#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
#############################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import torch
import torch.nn.functional as F
import math

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

class LanguageModel(torch.nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads):
        super(LanguageModel, self).__init__()

        self.startTokenIdx = 0
        self.endTokenIdx = 1
        self.unkTokenIdx = 2
        self.padTokenIdx = 3
        self.transTokenIdx = 4

        self.vocabSize = int(vocab_size)
        self.dModel = int(d_model)
        self.numLayers = int(num_layers)
        self.numHeads = int(num_heads)
        self.ffnSize = 4 * self.dModel
        self.dropoutProb = 0.1

        self.maxPositions = 2048
        self.sourceLossWeight = 0.2
        self.targetLossWeight = 1.0
        self.translationBeamSize = 5
        self.beamLengthPenalty = 0.7
        self.sourceSamplingTemperature = 0.9
        self.sourceTopK = 40
        self.sourceTopP = 0.95
        self.sourceRepetitionPenalty = 1.15

        self.tokenEmbedding = torch.nn.Embedding(
            self.vocabSize, self.dModel, padding_idx=self.padTokenIdx
        )
        self.positionEmbedding = torch.nn.Embedding(self.maxPositions, self.dModel)
        self.segmentEmbedding = torch.nn.Embedding(2, self.dModel)
        self.embeddingDropout = torch.nn.Dropout(self.dropoutProb)

        block = torch.nn.TransformerEncoderLayer(
            d_model=self.dModel,
            nhead=self.numHeads,
            dim_feedforward=self.ffnSize,
            dropout=self.dropoutProb,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(block, num_layers=self.numLayers)
        self.finalNorm = torch.nn.LayerNorm(self.dModel)
        self.outputProjection = torch.nn.Linear(self.dModel, self.vocabSize, bias=False)
        self.outputProjection.weight = self.tokenEmbedding.weight

    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        maxLen = max(len(s) for s in source)
        sentsPadded = [s + (maxLen - len(s)) * [self.padTokenIdx] for s in source]
        return torch.tensor(sentsPadded, dtype=torch.long, device=device)

    def _buildSegmentIds(self, tokens):
        transHits = (tokens == self.transTokenIdx).long()
        segmentIds = ((transHits.cumsum(dim=1) - transHits) > 0).long()
        segmentIds = segmentIds.masked_fill(tokens == self.padTokenIdx, 0)
        return segmentIds

    def _causalMask(self, seqLen, device):
        # Boolean attention mask: True means "blocked".
        mask = torch.ones((seqLen, seqLen), dtype=torch.bool, device=device)
        return torch.triu(mask, diagonal=1)

    def _encode(self, tokens):
        batch_size, seqLen = tokens.shape
        if seqLen > self.maxPositions:
            raise RuntimeError(
                "Sequence length {} exceeds positional limit {}".format(
                    seqLen, self.maxPositions
                )
            )

        device = tokens.device
        positions = torch.arange(seqLen, device=device).unsqueeze(0).expand(batch_size, seqLen)
        segmentIds = self._buildSegmentIds(tokens)

        x = self.tokenEmbedding(tokens)
        x = x + self.positionEmbedding(positions) + self.segmentEmbedding(segmentIds)
        x = self.embeddingDropout(x)

        padMask = tokens == self.padTokenIdx
        mask = self._causalMask(seqLen, device)
        h = self.transformer(x, mask=mask, src_key_padding_mask=padMask)
        h = self.finalNorm(h)
        return h, segmentIds

    def save(self, fileName):
        torch.save(self.state_dict(), fileName)

    def load(self, fileName):
        device = next(self.parameters()).device
        self.load_state_dict(torch.load(fileName, map_location=device))

    def forward(self, source):
        tokens = self.preparePaddedBatch(source)
        hidden, segmentIds = self._encode(tokens)
        logits = self.outputProjection(hidden[:, :-1, :])
        targets = tokens[:, 1:]

        perTokenLoss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)

        nonPad = (targets != self.padTokenIdx).float()
        targetSide = (segmentIds[:, 1:] == 1).float()
        weights = nonPad * (
            targetSide * self.targetLossWeight + (1.0 - targetSide) * self.sourceLossWeight
        )
        denom = weights.sum().clamp_min(1.0)
        return (perTokenLoss * weights).sum() / denom

    def _nextTokenLogProbs(self, seq):
        device = next(self.parameters()).device
        tokens = torch.tensor([seq], dtype=torch.long, device=device)
        hidden, _ = self._encode(tokens)
        logits = self.outputProjection(hidden[:, -1, :])
        if 0 <= self.unkTokenIdx < logits.size(-1):
            logits[:, self.unkTokenIdx] = float("-inf")
        return F.log_softmax(logits, dim=-1).squeeze(0)

    def _sampleToken(self, logProbs, seq):
        scores = logProbs.clone()

        if self.sourceRepetitionPenalty > 1.0 and len(seq) > 0:
            repeated = torch.tensor(sorted(set(seq)), device=scores.device, dtype=torch.long)
            scores[repeated] = scores[repeated] - math.log(self.sourceRepetitionPenalty)

        if self.sourceSamplingTemperature <= 0:
            return int(torch.argmax(scores).item())

        scores = scores / self.sourceSamplingTemperature

        topK = int(self.sourceTopK)
        if 0 < topK < scores.numel():
            cutoff = torch.topk(scores, k=topK).values[-1]
            scores = scores.masked_fill(scores < cutoff, float("-inf"))

        topP = float(self.sourceTopP)
        if 0.0 < topP < 1.0:
            sortedScores, sortedIdx = torch.sort(scores, descending=True)
            sortedProbs = torch.softmax(sortedScores, dim=-1)
            cumulative = torch.cumsum(sortedProbs, dim=-1)
            remove = cumulative > topP
            remove[0] = False
            sortedScores = sortedScores.masked_fill(remove, float("-inf"))
            filteredScores = torch.full_like(scores, float("-inf"))
            filteredScores[sortedIdx] = sortedScores
            scores = filteredScores

        probs = torch.softmax(scores, dim=-1)
        if not torch.isfinite(probs).all() or probs.sum().item() <= 0:
            return int(torch.argmax(logProbs).item())
        return int(torch.multinomial(probs, num_samples=1).item())

    def _rankScore(self, logProbSum, generatedLength):
        penalty = ((5.0 + generatedLength) / 6.0) ** self.beamLengthPenalty
        return logProbSum / penalty

    def _beamGenerate(self, prefix, limit):
        baseLen = len(prefix)
        beams = [(list(prefix), 0.0, False)]
        maxSteps = limit - baseLen

        for _ in range(maxSteps):
            candidates = []
            for seq, score, finished in beams:
                if finished:
                    candidates.append((seq, score, True))
                    continue

                logProbs = self._nextTokenLogProbs(seq)
                k = min(self.translationBeamSize, logProbs.size(0))
                values, indices = torch.topk(logProbs, k=k)
                for value, idx in zip(values.tolist(), indices.tolist()):
                    nextSeq = seq + [int(idx)]
                    isDone = idx == self.endTokenIdx
                    candidates.append((nextSeq, score + float(value), isDone))

            candidates.sort(
                key=lambda item: self._rankScore(
                    item[1], max(1, len(item[0]) - baseLen)
                ),
                reverse=True,
            )
            beams = candidates[: self.translationBeamSize]

            if all(done for _, _, done in beams):
                break

        finished = [b for b in beams if b[2]]
        bestPool = finished if finished else beams
        best = max(
            bestPool,
            key=lambda item: self._rankScore(item[1], max(1, len(item[0]) - baseLen)),
        )
        return best[0]

    def _greedyGenerate(self, prefix, limit, stopOnTrans=False, doSample=False):
        seq = list(prefix)
        while len(seq) < limit:
            if seq[-1] == self.endTokenIdx:
                break
            logProbs = self._nextTokenLogProbs(seq)
            nextToken = self._sampleToken(logProbs, seq) if doSample else int(torch.argmax(logProbs).item())
            seq.append(int(nextToken))
            if stopOnTrans and seq[-1] == self.transTokenIdx:
                break
        return seq

    def generate(self, prefix, limit=1000):
        if limit <= 0:
            return []

        if len(prefix) == 0:
            prefix = [self.startTokenIdx]
        elif len(prefix) >= limit:
            return list(prefix[:limit])

        useBeam = self.transTokenIdx in prefix
        with torch.no_grad():
            if useBeam:
                return self._beamGenerate(prefix, limit)

            # Two-stage decoding:
            # 1) continue the source side greedily until <TRANS> appears;
            # 2) once <TRANS> is present, decode target side with beam search.
            seq = self._greedyGenerate(prefix, limit, stopOnTrans=True, doSample=True)
            if len(seq) >= limit or seq[-1] == self.endTokenIdx or self.transTokenIdx not in seq:
                return seq
            return self._beamGenerate(seq, limit)
