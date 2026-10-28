# Results summary (auto-generated)

## E1 pooled NSE (capacity-normalised)

horizon          7      15     30
model                            
AR-Ridge      0.985  0.948  0.864
Climatology   0.979  0.939  0.857
HistGB        0.983  0.938  0.825
MLP           0.977  0.907  0.706
Persistence   0.976  0.927  0.808
RandomForest  0.985  0.947  0.835

## E1 pooled NSE by season

horizon                    7      15     30
reservoir model                            
ALL_DRY   AR-Ridge      0.995  0.976  0.914
          Climatology   0.990  0.961  0.894
          HistGB        0.996  0.980  0.923
          MLP           0.991  0.966  0.870
          Persistence   0.990  0.962  0.884
          RandomForest  0.996  0.983  0.927
ALL_NE    AR-Ridge      0.945  0.833  0.656
          Climatology   0.931  0.823  0.637
          HistGB        0.935  0.782  0.461
          MLP           0.923  0.672  0.085
          Persistence   0.916  0.766  0.466
          RandomForest  0.943  0.809  0.481
ALL_SW    AR-Ridge      0.993  0.970  0.885
          Climatology   0.993  0.973  0.912
          HistGB        0.993  0.969  0.871
          MLP           0.989  0.961  0.793
          Persistence   0.993  0.967  0.872
          RandomForest  0.994  0.975  0.889

## E1 pooled RMSE by season (fraction of capacity)

horizon                     7       15      30
reservoir model                               
ALL_DRY   AR-Ridge      0.0172  0.0376  0.0718
          Climatology   0.0244  0.0480  0.0798
          HistGB        0.0151  0.0342  0.0680
          MLP           0.0227  0.0451  0.0882
          Persistence   0.0240  0.0475  0.0835
          RandomForest  0.0146  0.0320  0.0663
ALL_NE    AR-Ridge      0.0529  0.0924  0.1347
          Climatology   0.0591  0.0951  0.1384
          HistGB        0.0575  0.1058  0.1685
          MLP           0.0625  0.1296  0.2196
          Persistence   0.0653  0.1094  0.1678
          RandomForest  0.0537  0.0989  0.1655
ALL_SW    AR-Ridge      0.0170  0.0353  0.0679
          Climatology   0.0172  0.0340  0.0596
          HistGB        0.0175  0.0361  0.0719
          MLP           0.0217  0.0405  0.0913
          Persistence   0.0179  0.0371  0.0717
          RandomForest  0.0168  0.0325  0.0669

## E1 RMSE (mcft) per reservoir

horizon                          7      15     30
reservoir       model                            
ALL_DRY         AR-Ridge        0.0    0.0    0.1
                Climatology     0.0    0.0    0.1
                HistGB          0.0    0.0    0.1
                MLP             0.0    0.0    0.1
                Persistence     0.0    0.0    0.1
                RandomForest    0.0    0.0    0.1
ALL_NE          AR-Ridge        0.1    0.1    0.1
                Climatology     0.1    0.1    0.1
                HistGB          0.1    0.1    0.2
                MLP             0.1    0.1    0.2
                Persistence     0.1    0.1    0.2
                RandomForest    0.1    0.1    0.2
ALL_SW          AR-Ridge        0.0    0.0    0.1
                Climatology     0.0    0.0    0.1
                HistGB          0.0    0.0    0.1
                MLP             0.0    0.0    0.1
                Persistence     0.0    0.0    0.1
                RandomForest    0.0    0.0    0.1
CHEMBARAMBAKKAM AR-Ridge       82.9  158.8  285.0
                Climatology    86.8  149.5  240.9
                HistGB         90.7  172.5  363.5
                MLP           110.2  214.4  506.0
                Persistence    99.7  185.1  316.6
                RandomForest   79.3  156.1  350.4
CHOLAVARAM      AR-Ridge       31.6   55.7   84.0
                Climatology    35.9   59.2   86.3
                HistGB         31.3   58.4   93.2
                MLP            35.1   70.3  118.3
                Persistence    37.3   61.1   90.9
                RandomForest   31.3   55.4   91.1
POONDI          AR-Ridge      111.9  208.3  317.2
                Climatology   136.8  239.4  366.3
                HistGB        117.3  231.4  369.0
                MLP           132.2  256.2  401.6
                Persistence   141.9  249.7  399.1
                RandomForest  109.3  214.4  357.9
REDHILLS        AR-Ridge       74.0  144.2  266.2
                Climatology    85.1  152.5  256.0
                HistGB         86.9  157.9  257.8
                MLP           103.8  233.5  403.0
                Persistence   100.2  191.4  339.7
                RandomForest   78.5  146.7  251.6

## E1 Diebold-Mariano vs persistence (HistGB)

 horizon  model       reservoir  DM_stat  p_value
       7 HistGB          POONDI   1.6319   0.1029
       7 HistGB      CHOLAVARAM   1.1549   0.2483
       7 HistGB        REDHILLS   1.2041   0.2288
       7 HistGB CHEMBARAMBAKKAM   0.7491   0.4539
      15 HistGB          POONDI   0.6162   0.5379
      15 HistGB      CHOLAVARAM   0.3143   0.7533
      15 HistGB        REDHILLS   1.2406   0.2149
      15 HistGB CHEMBARAMBAKKAM   0.4431   0.6577
      30 HistGB          POONDI   0.5705   0.5684
      30 HistGB      CHOLAVARAM  -0.1027   0.9182
      30 HistGB        REDHILLS   1.2647   0.2062
      30 HistGB CHEMBARAMBAKKAM  -0.5868   0.5574

## E2 90 % intervals (ALL; MPIW = fraction of capacity)

 horizon        model       method  PICP  PICP_NEmonsoon  MPIW  Winkler
       7       HistGB     Split-CP 0.919           0.720 0.084  392.399
       7       HistGB  Mondrian-CP 0.947           0.941 0.093  334.911
       7       HistGB          ACI 0.896           0.658 0.074  388.406
       7 RandomForest     Split-CP 0.920           0.730 0.076  359.761
       7 RandomForest  Mondrian-CP 0.950           0.944 0.090  320.149
       7 RandomForest          ACI 0.896           0.676 0.076  373.191
       7          MLP     Split-CP 0.940           0.810 0.115  452.607
       7          MLP  Mondrian-CP 0.964           0.983 0.142  457.710
       7          MLP          ACI 0.897           0.700 0.091  451.499
       7       HistGB Quantile-reg 0.776           0.786 0.040  271.782
      15       HistGB     Split-CP 0.918           0.718 0.174  724.137
      15       HistGB  Mondrian-CP 0.928           0.942 0.175  633.466
      15       HistGB          ACI 0.897           0.662 0.198  823.685
      15 RandomForest     Split-CP 0.934           0.780 0.175  681.430
      15 RandomForest  Mondrian-CP 0.953           0.955 0.174  587.083
      15 RandomForest          ACI 0.898           0.647 0.163  732.361
      15          MLP     Split-CP 0.946           0.805 0.246  902.181
      15          MLP  Mondrian-CP 0.955           0.955 0.231  787.489
      15          MLP          ACI 0.904           0.661 0.201  979.020
      15       HistGB Quantile-reg 0.702           0.738 0.079  594.764
      30       HistGB     Split-CP 0.944           0.802 0.345 1186.004
      30       HistGB  Mondrian-CP 0.934           1.000 0.376 1261.686
      30       HistGB          ACI 0.898           0.684 0.342 1412.647
      30 RandomForest     Split-CP 0.943           0.806 0.346 1173.733
      30 RandomForest  Mondrian-CP 0.950           1.000 0.379 1248.733
      30 RandomForest          ACI 0.899           0.687 0.343 1388.388
      30          MLP     Split-CP 0.931           0.749 0.406 1543.260
      30          MLP  Mondrian-CP 0.948           0.967 0.413 1372.303
      30          MLP          ACI 0.900           0.680 0.388 1780.662
      30       HistGB Quantile-reg 0.665           0.682 0.141 1046.632

## E3 transfer NSE

horizon                              7      15     30
reservoir       setting                              
CHEMBARAMBAKKAM LORO (unseen)     0.988  0.950  0.787
                Local (own data)  0.984  0.939  0.613
                Persistence       0.985  0.948  0.848
                Pooled (all 4)    0.987  0.955  0.800
CHOLAVARAM      LORO (unseen)     0.943  0.818  0.543
                Local (own data)  0.924  0.791  0.518
                Persistence       0.929  0.812  0.587
                Pooled (all 4)    0.950  0.828  0.566
POONDI          LORO (unseen)     0.953  0.824  0.601
                Local (own data)  0.953  0.843  0.585
                Persistence       0.945  0.831  0.574
                Pooled (all 4)    0.963  0.855  0.636
REDHILLS        LORO (unseen)     0.992  0.972  0.924
                Local (own data)  0.989  0.969  0.898
                Persistence       0.989  0.961  0.878
                Pooled (all 4)    0.992  0.974  0.930

## E4 satellite sensitivity, pooled NSE (SIMULATED observations)

                          scenario  noise_frac  horizon  mean   std
                       Daily gauge        0.00        7 0.983   NaN
                       Daily gauge        0.00       15 0.938   NaN
                       Daily gauge        0.00       30 0.825   NaN
Optical only (5 d, cloudy monsoon)        0.00        7 0.942 0.009
Optical only (5 d, cloudy monsoon)        0.00       15 0.893 0.008
Optical only (5 d, cloudy monsoon)        0.00       30 0.760 0.025
Optical only (5 d, cloudy monsoon)        0.02        7 0.930 0.008
Optical only (5 d, cloudy monsoon)        0.02       15 0.876 0.004
Optical only (5 d, cloudy monsoon)        0.02       30 0.751 0.009
Optical only (5 d, cloudy monsoon)        0.05        7 0.902 0.004
Optical only (5 d, cloudy monsoon)        0.05       15 0.846 0.013
Optical only (5 d, cloudy monsoon)        0.05       30 0.714 0.046
Optical only (5 d, cloudy monsoon)        0.10        7 0.850 0.008
Optical only (5 d, cloudy monsoon)        0.10       15 0.799 0.006
Optical only (5 d, cloudy monsoon)        0.10       30 0.679 0.021
       SAR+optical (5 d + S1 12 d)        0.00        7 0.969 0.001
       SAR+optical (5 d + S1 12 d)        0.00       15 0.926 0.002
       SAR+optical (5 d + S1 12 d)        0.00       30 0.802 0.004
       SAR+optical (5 d + S1 12 d)        0.02        7 0.957 0.001
       SAR+optical (5 d + S1 12 d)        0.02       15 0.908 0.001
       SAR+optical (5 d + S1 12 d)        0.02       30 0.781 0.003
       SAR+optical (5 d + S1 12 d)        0.05        7 0.936 0.003
       SAR+optical (5 d + S1 12 d)        0.05       15 0.882 0.002
       SAR+optical (5 d + S1 12 d)        0.05       30 0.758 0.002
       SAR+optical (5 d + S1 12 d)        0.10        7 0.893 0.005
       SAR+optical (5 d + S1 12 d)        0.10       15 0.832 0.007
       SAR+optical (5 d + S1 12 d)        0.10       30 0.714 0.002
