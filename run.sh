cd build 
rm -rf * 
cmake -DGPU_BACKEND=HIP .. 
make
cd .. 
./build/gauss_seidel 