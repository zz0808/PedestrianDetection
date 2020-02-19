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
//	DescriptorDim = svm->getVarCount();//特征向量的维数，即HOG描述子的维数
//	cv::Mat supportVector = svm->getSupportVectors();//支持向量的个数
//	int supportVectorNum = supportVector.rows;
//
//	std::vector<float> svm_alpha;
//	std::vector<float> svm_svidx;
//	float  svm_rho;
//	svm_rho = svm->getDecisionFunction(0, svm_alpha, svm_svidx);
//
//	cv::Mat alphaMat = cv::Mat::zeros(1, supportVectorNum, CV_32FC1); //alpha 向量，长度等于支持向量个数
//	cv::Mat supportVectorMat = cv::Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1); //支持向量矩阵
//	cv::Mat resultMat = cv::Mat::zeros(1, DescriptorDim, CV_32FC1); // alpha 向量乘以支持向量矩阵的结果
//
//	  //将支持向量的数据复制到supportVectorMat矩阵中
//
//	supportVectorMat = supportVector;
//
//	//将alpha向量的数据复制到alphaMat中
//	///double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量
//	for (int i = 0; i < supportVectorNum; i++)
//	{
//		alphaMat.at<float>(0, i) = svm_alpha[i];
//	}
//
//	//计算-(alphaMat * supportVectorMat),结果放到resultMat中
//	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);
//	resultMat = -1 * alphaMat * supportVectorMat;
//
//	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
//	std::vector<float> myDetector;
//	//将resultMat中的数据复制到数组myDetector中
//	for (int i = 0; i < DescriptorDim; i++)
//	{
//		myDetector.push_back(resultMat.at<float>(0, i));
//	}
//
//	//最后添加偏移量rho，得到检测子
//   /// myDetector.push_back(svm.get_rho());
//	myDetector.push_back(svm_rho);
//	std::cout << "检测子维数：" << myDetector.size() << std::endl;
//
//	//设置HOGDescriptor的检测子
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
//		//对负样本原图进行多尺度检测，检测出的都是误报
//		myHOG.detectMultiScale(img, found, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);
//		//遍历从图像中检测出来的矩形框，得到hard example
//		size_t i, j;
//
//		//遍历从图像中检测出来的矩形框，得到hard example
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
//			//从原图上截取矩形框大小的图片
//			cv::Mat imgROI = img(cv::Rect(r.x, r.y, r.width, r.height));
//
//			//将剪裁出来的图片缩放为64*128大小
//			cv::resize(imgROI, imgROI, cv::Size(64, 128));
//
//			sprintf_s(saveName, 256, ".//img_dir//hard//hard_%06d.png", ++HardSamCount);
//
//			cv::imwrite(saveName, imgROI);
//
//			//保存裁剪得到的图片名称到txt文件，换行分隔
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
