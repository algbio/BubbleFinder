# BubbleFinder

To build:
```
mkdir build
cd build
cmake .. 
make 
```

To run BubbleFinder:
```
cd build 
./BubbleFinder -g {graphPath} [--gfa] -o {-, outputPath} [--superbubbles | --snarls] -j {threadsNumber} -m {stack size per thread in bytes}
```

To run brute-force tester for snarls
```
cd build 
./snarls_brute input_path
```