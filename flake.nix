{
  description = "xarray and dask wrapped scipy";

  inputs.nixpkgs.url = "github:nixos/nixpkgs?ref=master";
  inputs.utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, utils }: {
    overlays = {
      default = final: prev: {
        pythonPackagesOverrides = (prev.pythonPackagesOverrides or []) ++ [
          (self: super: {
            xarray-scipy = self.callPackage ./. {};
            # xarray = super.xarray.overridePythonAttrs(oldAttrs: rec {
            #   version = src.rev;
            #   src = final.fetchFromGitHub {
            #     owner = "pydata";
            #     repo = "xarray";
            #     rev = "8f6873026587f30f0ac10af740ebe947797208a1";
            #     hash = "sha256-CTvVTOs2zOqX1NJB4LUP6EDY8RZcrOQaMvvhZktCl+k=";
            #   };
            # });
          })
        ];
      };
    };
  } // (utils.lib.eachSystem [ "x86_64-linux" ] (system: let
    # Our own overlay does not get applied to nixpkgs because that would lead to
    # an infinite recursion. Therefore, we need to import nixpkgs and apply it ourselves.
    pkgs = import nixpkgs {
      inherit system;
      overlays = [
          self.overlays.default
      ];
    };
    python = let
      composeOverlays = pkgs.lib.foldl' pkgs.lib.composeExtensions (self: super: { });
      self = pkgs.python3.override {
        inherit self;
        packageOverrides = composeOverlays pkgs.pythonPackagesOverrides;
      };
    in self; 
  in rec {
    packages = rec {
      # Development environment that includes our package, its dependencies
      # and additional dev inputs.
      devEnv = python.withPackages(_: pkg.allInputs);
      pkg = python.pkgs.xarray-scipy;
      default = pkg;
    };

    devShells = {
      default = pkgs.mkShell {
        nativeBuildInputs = [
          packages.devEnv
        ];
        shellHook = ''
          export PYTHONPATH=$(readlink -f .):$PYTHONPATH
        '';
      };
    };
  }));
}
