# C3Net-for-building-extraction

This is a PyTorch implementation for our paper "[Context-content collaborative network for building extraction from high-resolution imagery](https://www.sciencedirect.com/science/article/abs/pii/S0950705123000333)" has been published on Knowledge-Based System by Maoguo Gong, Tongfei Liu, Mingyang Zhang, Qingfu Zhang, Di Lu, Hanhong Zheng, Fenling Jiang.

## Requirements
>python  
pytorch  
opencv-python=4.1.0.25  
scikit-image  
scikit-learn   
tqdm  

## Usage
### Train
Load the train and test(val) data path  
run: python train.py  

### Test
1. Load the model(pth)  
2. Load the test data path  
run: python test.py  

### Get results (Visual and Quantitative)
**Visual result:** ./data/EastAsia/test/results  
**Quantitative result:** ./test_acc.txt   

## Citation
If you find our work useful for your research, please consider citing our paper:  
```
@article{gong2023context,
  title={Context-content collaborative network for building extraction from high-resolution imagery},
  author={Gong, Maoguo and Liu, Tongfei and Zhang, Mingyang and Zhang, Qingfu and Lu, Di and Zheng, Hanhong and Jiang, Fenlong},
  journal={Knowledge-Based Systems},
  pages={110283},
  year={2023},
  publisher={Elsevier}
}
```

## Contact us 
If you have any problme when running the code, please do not hesitate to contact us. Thanks.  
E-mail: liutongfei_home@hotmail.com  
Date: March 5, 2023  
