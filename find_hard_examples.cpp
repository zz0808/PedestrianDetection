//// This function in order to find all hard examples
//
//#include <iostream>
//#include <fstream>
//#include <opencv2/opencv.hpp>
//#include <vector>
//#include "detection.h"
//
//int HardSamCount = 0;
//
//int main() {
//
//	std::ifstream fin(NegSamListFile);
//	std::ofstream fout(HardSamListFile, std::ios::trunc);
//	std::string ImgName;
//	char saveName[256];
//
//	int DescriptorDim;
//
//	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(SvmListFile);
//
//	std::cout << "loaded SVM_HOG.xml file" << std::endl;
//
//	DescriptorDim = svm->getVarCount();//����������ά������HOG�����ӵ�ά��
//	cv::Mat supportVector = svm->getSupportVectors();//֧�������ĸ���
//	int supportVectorNum = supportVector.rows;
//
//	std::vector<float> svm_alpha;
//	std::vector<float> svm_svidx;
//	float  svm_rho;
//	svm_rho = svm->getDecisionFunction(0, svm_alpha, svm_svidx);
//
//	cv::Mat alphaMat = cv::Mat::zeros(1, supportVectorNum, CV_32FC1); //alpha ���������ȵ���֧����������
//	cv::Mat supportVectorMat = cv::Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1); //֧����������
//	cv::Mat resultMat = cv::Mat::zeros(1, DescriptorDim, CV_32FC1); // alpha ��������֧����������Ľ��
//
//	  //��֧�����������ݸ��Ƶ�supportVectorMat������
//
//	supportVectorMat = supportVector;
//
//	//��alpha���������ݸ��Ƶ�alphaMat��
//	///double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����
//	for (int i = 0; i < supportVectorNum; i++)
//	{
//		alphaMat.at<float>(0, i) = svm_alpha[i];
//	}
//
//	//����-(alphaMat * supportVectorMat),����ŵ�resultMat��
//	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);
//	resultMat = -1 * alphaMat * supportVectorMat;
//
//	//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����
//	std::vector<float> myDetector;
//	//��resultMat�е����ݸ��Ƶ�����myDetector��
//	for (int i = 0; i < DescriptorDim; i++)
//	{
//		myDetector.push_back(resultMat.at<float>(0, i));
//	}
//
//	//������ƫ����rho���õ������
//   /// myDetector.push_back(svm.get_rho());
//	myDetector.push_back(svm_rho);
//	std::cout << "�����ά����" << myDetector.size() << std::endl;
//
//	//����HOGDescriptor�ļ����
//	cv::HOGDescriptor myHOG;
//	myHOG.setSVMDetector(myDetector);
//
//	cv::Mat image;
//
//	while (std::getline(fin, ImgName)) {
//
//		std::cout << "now processing" << ImgName << std::endl;
//		image = cv::imread(ImgName);
//
//		cv::Mat img = image.clone();
//
//		std::vector<cv::Rect> found, found_filtered;
//		//�Ը�����ԭͼ���ж�߶ȼ�⣬�����Ķ�����
//		myHOG.detectMultiScale(img, found, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);
//		//������ͼ���м������ľ��ο򣬵õ�hard example
//		size_t i, j;
//
//		//������ͼ���м������ľ��ο򣬵õ�hard example
//		for (i = 0; i < found.size(); i++) {
//
//			cv::Rect r = found[i];
//			for (j = 0; j < found.size(); j++)
//				if (j != i && (r & found[j]) == r)
//					break;
//			if (j == found.size())
//				found_filtered.push_back(r);
//		}
//
//		for (i = 0; i < found_filtered.size(); i++) {
//
//			cv::Rect r = found_filtered[i];
//
//			if (r.x < 0)
//				r.x = 0;
//			if (r.y < 0)
//				r.y = 0;
//			if (r.x + r.width > img.cols)
//				r.width = img.cols - r.x;
//			if (r.y + r.height > img.rows)
//				r.height = img.rows - r.y;
//
//			//��ԭͼ�Ͻ�ȡ���ο��С��ͼƬ
//			cv::Mat imgROI = img(cv::Rect(r.x, r.y, r.width, r.height));
//
//			//�����ó�����ͼƬ����Ϊ64*128��С
//			cv::resize(imgROI, imgROI, cv::Size(64, 128));
//
//			sprintf_s(saveName, 256, ".//img_dir//hard//hard_%06d.png", ++HardSamCount);
//
//			cv::imwrite(saveName, imgROI);
//
//			//����ü��õ���ͼƬ���Ƶ�txt�ļ������зָ�
//			fout << ".//img_dir//hard//hard_" << std::setw(6) << std::setfill('0') << HardSamCount << ".png" << std::endl;
//		}
//	}
//
//	std::cout << "Total hard examples : " << HardSamCount << std::endl;
//
//	fin.close();
//	fout.close();
//
//	return 0;
//}
//
