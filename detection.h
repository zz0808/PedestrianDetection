#pragma once

// ѵ��
#define TRAIN false //�Ƿ����ѵ��,true��ʾ����ѵ����false��ʾ��ȡxml�ļ��е�SVMģ��
#define HARDNEG false //�Ƿ�ʹ��hardneg��true��ʾʹ��

#define DEALIMAGE false 
#define CROP false  // true ��ʾ�Ը�����ͼ��ü�

#define PosSamNO 2416  //ԭʼ��������2416 
#define cropNegNum 12180 // ÿ������ü���10�ţ���12140��
#define HardExampleNO 169 // ����
// δ���������������
#define NEGTRAINFile ".//train//neg"
#define POSTRAINFile ".//train//pos"


// ------------------------------�����޸�----------------------------------------------------//
#define NegativeImageList ".//data//sample_neg.txt"
#define PostiveImageList ".//data//sample_pos.txt"

//������ͼƬ���ļ����б�
#define PosSamListFile ".//data//pos.txt"
//������ͼƬ���ļ����б�
#define NegSamListFile ".//data//neg.txt"
// hard �������ļ����б�
#define HardSamListFile ".//data//hard.txt" 
//ѵ����HOG������svm����ʱʹ��
#define SvmListFile ".//data//svm.xml"
//ѵ����HOG������resultMat���
#define HogDetectorListFile ".//data//HOGDetectorForOpenCV7.txt"


