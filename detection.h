#pragma once

// 训练
#define TRAIN false //是否进行训练,true表示重新训练，false表示读取xml文件中的SVM模型
#define HARDNEG false //是否使用hardneg，true表示使用

#define DEALIMAGE false 
#define CROP false  // true 表示对负样本图像裁剪

#define PosSamNO 2416  //原始正样本数2416 
#define cropNegNum 12180 // 每张随机裁剪出10张，共12140张
#define HardExampleNO 169 // 难例
// 未处理的正、负样本
#define NEGTRAINFile ".//train//neg"
#define POSTRAINFile ".//train//pos"


// ------------------------------不需修改----------------------------------------------------//
#define NegativeImageList ".//data//sample_neg.txt"
#define PostiveImageList ".//data//sample_pos.txt"

//正样本图片的文件名列表
#define PosSamListFile ".//data//pos.txt"
//负样本图片的文件名列表
#define NegSamListFile ".//data//neg.txt"
// hard 样例的文件名列表
#define HardSamListFile ".//data//hard.txt" 
//训练的HOG特征，svm分类时使用
#define SvmListFile ".//data//svm.xml"
//训练的HOG特征，resultMat结果
#define HogDetectorListFile ".//data//HOGDetectorForOpenCV7.txt"


