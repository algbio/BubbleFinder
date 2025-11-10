To build:
```
mkdir build
cd build
cmake .. 
make 
```

To run:
```
cd build 
./BubbleFinder -g {graphPath} [--gfa] -o {-, outputPath} -j {threadsNumber} -m {stack size per thread in bytes}
```
