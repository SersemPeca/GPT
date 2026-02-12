{
  description = "Python AI Devshell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (
        pkgs:
        let
          pythonEnv = pkgs.python3.withPackages (
            pypkgs: with pypkgs; [
              (torch.override { rocmSupport = true; })
              python
              ipython
              numpy
              scipy
              tqdm
              docopt
              nltk
              pkgs.nltk-data.punkt
              # torchvision
              scikit-learn
              matplotlib
              pdftotext
              pip
              jupyter
              notebook
              ipykernel

              pkgs.nltk-data.punkt
              pkgs.nltk-data.punkt-tab
            ]
          );
        in
        {
          default = pkgs.mkShell {
            packages = [
              pythonEnv
              pkgs.poppler-utils
            ];

            shellHook = ''
              # Point NLTK to local project directory
              # export NLTK_DATA="$PWD/nltk_data"
              export NLTK_DATA="$PWD/nltk_data:${pkgs.nltk-data.punkt}:${pkgs.nltk-data.punkt-tab}"
              echo "NLTK_DATA set to $NLTK_DATA"
              fish
            '';
          };
        }
      );
    };
}
