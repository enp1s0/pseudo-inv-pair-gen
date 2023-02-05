# pseudo-inv matrix pair gen

Generating a pair of rand matrix and pseudo-inv manrix.

## Build
```bash
git clone git@github.com:enp1s0/pseud-inv-pair-gen.git
cd pseud-inv-pair-gen
make
```

## Run
```
./pseudo-inv-pair [N (N x N)] [dtype: fp32/fp64] [seed]
```

It generates two matrices:

- matrix A (e.g. `bebc9d0-dp-m2-n2-seed0.matrix`)
- pseudo-inv matrix A^t (e.g. `bebc9d0-inv-dp-m2-n2-seed0.matrix`)

where A^t A = I.

## Load matrices

- See [matfile](https://github.com/enp1s0/matfile).

## License
MIT
