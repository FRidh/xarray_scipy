{ buildPythonPackage

, black
, flit
, pylint

, pytest
, dask

, scipy
, xarray

, notebook

}:

let
  # Development tools used during package build
  nativeBuildInputs = [
    black
    flit
    pylint
  ];

  # Run-time Python dependencies
  propagatedBuildInputs = [
    scipy
    xarray
  ];

  # Test-dependencies
  checkInputs = [
    dask
    pytest
  ];

  # Other dev dependencies not used during build
  devInputs = [
    notebook
  ];

  allInputs = nativeBuildInputs ++ propagatedBuildInputs ++ checkInputs ++ devInputs;

  pkg = buildPythonPackage {
    pname = "xarray-scipy";
    version = "dev";
    format = "pyproject";

    src = ./.;

    inherit nativeBuildInputs propagatedBuildInputs checkInputs;

    preBuild = ''
      echo "Checking for errors with pylint..."
      pylint -E xarray_scipy
    '';

    postInstall = ''
      echo "Checking formatting..."
      black --check xarray_scipy
    '';

    checkPhase = ''
      pytest tests
    '';

    doCheck = true;

    shellHook = ''
      export PYTHONPATH="$(pwd):"$PYTHONPATH""
    '';

    outputs = [
      "out"
    ];

    passthru = {
      inherit allInputs;
    };
  };

in pkg
