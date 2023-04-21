#### 3. Single-Camera Multi-Target Tracking

Prepare ReID results.
```
./dataspace/AICITY_test      
  ├── aic22_1_test_infer_v2_Convnext.pkl
  ├── aic22_1_test_infer_v2_HR48_eps.pkl
  ├── aic22_1_test_infer_v2_R50.pkl
  ├── aic22_1_test_infer_v2_Res2Net200.pkl
  ├── aic22_1_test_infer_v2_ResNext101.pkl
  ├── gen_detection_feat.py
```

Prepare the extracted images of test set by ffmpeg.
```
./dataset/CityFlowV2/AICITY/test/
  ├── c041
    ├── img1
      ├── 0001.jpg
      ├── 0002.jpg
      ...
    ├── seqinfo.ini
  ├── c042
  ├── c043
  ├── c044
  ├── c045
  ├── c046
```

Generate detection and ReID feature in the format of tracking input.
```
cd ./dataspace/AICITY_test
python gen_detection_feat.py
cd ../..
```

Run tracking.
```
python run_aicty.py AICITY test --dir_save scmt
python stat_occlusion_scmt.py scmt
```

The SCMT tracking results will be generated in 'scmt' 

