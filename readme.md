#### 使用SVM+HOG实现行人检测



+ 数据集：INRIAPerson

+ 正样本：INRIAPerson/96x160H96/Train/pos，一共2416张；负样本：INRIAPerson/Train/neg，1218张，将负样本每张随机剪裁成10张，共12180张，正样本从中间剪裁，大小都是64*128。使用该数据集可能会报libpng的错误，运行test.py脚本可以解决

+ 只需在detection.h修改样本个数等信息

+ 首先将正、负样本进行训练，对于被识别成有行人的负样本，标记为hard examples，进行第二轮训练

+ 正、负样本在train/pos|neg文件夹

  ![Test](.\Test.png)