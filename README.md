# WaveForM

This is a PyTorch implementation of the paper: WaveForM: Graph Enhanced Wavelet Learning for Long Sequence Forecasting of Multivariate Time Series, published in AAAI 2023.

## Requirements

To run the code, you need to install `Python(>=3.9.12)` and `PyTorch(>=1.11.0)` at least. The full requirements are specified in `requirements.txt`.

The `pytorch_wavelets` package should be installed following the instructions of [pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets).

## Data

The Electricity, Solar-energy and Traffic datasets can be downloaded from [multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data).

The Weather datasets can be downloaded from [Autoformer](https://github.com/thuml/Autoformer).

The Temperature datasets (asos) can be downloaded from [spacetimeformer](https://github.com/QData/spacetimeformer).

You should put the `xx.csv` file into the directory with the datasets' name in `dataset` directory.

For example, the proper file structure should be like:
```
dataset
|-- electricity
| |-- electricity.csv
|--solar
| |-- solar.csv
|--temperature
| |-- temperature.csv
|--traffic
| |-- traffic.csv
|--weather
| |-- weather.csv
```

## Running the code

The running script is `run.sh`, where you can change any arguments which have been declared in `run.py`.

## Contact

If you have any questions, you can raise an issue or send an e-mail to yfh@bit.edu.cn.

## Acknowledgement

Thanks to the following repos for their codes and datasets.

[https://github.com/fbcotter/pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets)

[https://github.com/thuml/Autoformer](https://github.com/thuml/Autoformer)

[https://github.com/nnzhan/MTGNN](https://github.com/nnzhan/MTGNN)

[https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data)


