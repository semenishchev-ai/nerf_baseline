for ext in raymarching gridencoder shencoder freqencoder; do
    pip install ./$ext --no-build-isolation
done

# turned off by default, very slow to compile, and performance is not good enough.
#pip install ./ffmlp --no-build-isolation