#pragma once

int deal_image() {

	std::ofstream fout;
	// 处理训练负样本
	std::vector<cv::String> NegFileList;
	cv::glob(NEGTRAINFile, NegFileList);
	// 处理训练正样本
	std::vector<cv::String> PosFileList;
	cv::glob(POSTRAINFile, PosFileList);

	// 创建data文件夹和txt文件
	if (NegFileList.size() == 0 || PosFileList.size() == 0) {
		std::cout << "训练文件不存在" << std::endl;
		return -1;
	}

	if (_access(".//data", 0) != 0) {
		int flag = _mkdir(".//data");
		if (flag != 0) {
			std::cout << "创建文件失败" << std::endl;
			return -1;
		}
	}

	if (_access(".//img_dir", 0) != 0) {
		int flag = _mkdir(".//img_dir");
		if (flag == 0) {
			int flag1 = _mkdir(".//img_dir//pos");
			flag1 = _mkdir(".//img_dir//neg");
			flag1 = _mkdir(".//img_dir//hard");
		}
	}

	fout.open(NegativeImageList, std::ios::trunc);

	for (size_t i = 0; i < NegFileList.size(); i++) {
		fout << NegFileList[i].c_str() << std::endl;
	}
	fout.close();

	fout.open(PostiveImageList, std::ios::trunc);
	
	for (size_t i = 0; i < PosFileList.size(); i++) {
		fout << PosFileList[i].c_str() << std::endl;
	}

	fout.close();


	return 0;
}


int crop_image() {

	std::ifstream fin;
	std::ofstream fout;
	std::string ImgName;
	cv::Mat image;
	int CropNegNumCount = 0;
	char FileName[256];
	
	// 剪裁负样本
	fin.open(NegativeImageList);

	fout.open(".//data//neg.txt", std::ios::trunc);
	// 每张负样本随机裁剪出10张
	for (int i = 0; i < cropNegNum && std::getline(fin, ImgName); i++) {

		image = cv::imread(ImgName);
		// 裁剪出的图片大小是64*128，所以被裁剪的图片不小于64*128
		if (image.cols > 64 && image.rows > 128) {
			srand((unsigned)time(NULL));
			for (int j = 0; j < 10; j++) {
				int x = rand() % (image.cols - 64);
				int y = rand() % (image.rows - 128);
				cv::Mat ImageROI = image(cv::Rect(x, y, 64, 128));
				sprintf_s(FileName, 256, ".//img_dir//neg//neg_%06d.png", ++CropNegNumCount);
				cv::imwrite(FileName, ImageROI);
				fout << ".//img_dir//neg//neg_" << std::setw(6) << std::setfill('0') << std::to_string(CropNegNumCount) << ".png" << std::endl;
				std::cout << "已处理第" << CropNegNumCount << "张负样本" << std::endl;
			}
		}

	}

	std::cout << "一共裁剪出负样本（张）: " << CropNegNumCount << std::endl;
	CropNegNumCount = 0;

	fin.close();
	fout.close();

	// 剪裁正样本
	fout.open(".//data//pos.txt", std::ios::trunc);
	fin.open(PostiveImageList);

	for (int i = 0; i < PosSamNO && std::getline(fin, ImgName); i++) {

		image = cv::imread(ImgName);

		if (image.cols > 64 && image.rows > 128) {
			int x = image.cols / 2 - 32; // 从中间剪裁
			int y = image.rows / 2 - 64;
			cv::Mat ImageROI = image(cv::Rect(x, y, 64, 128));
			sprintf_s(FileName, 256, ".//img_dir//pos//pos_%06d.png", ++CropNegNumCount);
			cv::imwrite(FileName, ImageROI);
			fout << ".//img_dir//pos//pos_" << std::setw(6) << std::setfill('0') << std::to_string(CropNegNumCount) << ".png" << std::endl;
			std::cout << "已处理第" << CropNegNumCount << "张正样本" << std::endl;

		}
	}
	std::cout << "一共裁剪出正样本（张）: " << CropNegNumCount << std::endl;

	fin.close();
	fout.close();

	return 0;
}


// 训练svm
void svm_train(cv::HOGDescriptor& hog, int& DescriptorDim, cv::Ptr<cv::ml::SVM>& svm){


	std::string imgName; // 图片名
	std::ifstream finNeg(NegSamListFile); // 负样本文件名
	std::ifstream finPos(PosSamListFile); // 正样本文件名
	std::ifstream finHard(HardSamListFile); // 难例文件名

	// 判断文件是否存在
	if (HARDNEG) {
		if (!finNeg || !finPos || !finHard) {
			std::cout << "Pos/Neg/hardNeg imglist reading failed..." << std::endl;
			return;
		}
	}
	else {
		if (!finNeg || !finPos) {
			std::cout << "Pos/Neg imglist reading failed..." << std::endl;
			return;
		}
	}

	cv::Mat sampleFeatureMat;
	cv::Mat sampleLableMat;

	// 处理正样本，生成HOG描述子
	for (int num = 0; num < PosSamNO && std::getline(finPos, imgName); num++) {
		std::cout << "Now processing original positive image: " << imgName << std::endl;
		cv::Mat image = cv::imread(imgName);

		std::vector<float> descriptor;
		hog.compute(image, descriptor, cv::Size(8, 8)); // 计算HOG描述子，窗口步长为8*8

		if (num == 0) {
			DescriptorDim = descriptor.size();
			//std::cout << "DescriptorDim: " << DescriptorDim << std::endl;
			sampleFeatureMat = cv::Mat::zeros(PosSamNO + cropNegNum + HardExampleNO, DescriptorDim, CV_32FC1);
			sampleLableMat = cv::Mat::zeros(PosSamNO + cropNegNum + HardExampleNO, 1, CV_32SC1);
		}

		for (int i = 0; i < DescriptorDim; i++) {
			sampleFeatureMat.at<float>(num, i) = descriptor[i];
			sampleLableMat.at<int>(num, 0) = 1; // 1： 正样本
		}
	}
 
	// 处理负样本，生成HOG描述子
	for (int num = 0; num < cropNegNum && std::getline(finNeg, imgName); num++) {
		std::cout << "Now processing original negative image: " << imgName << std::endl;
		cv::Mat image = cv::imread(imgName);

		std::vector<float> descriptor;
		hog.compute(image, descriptor, cv::Size(8, 8)); // 计算HOG描述子，窗口步长为8*8

		//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
		for (int i = 0; i < DescriptorDim; i++) {
			sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptor[i];
			sampleLableMat.at<int>(num + PosSamNO, 0) = -1; // -1: 表示负样本，不含行人
		}
	}

	// 处理难例， 生成HOG描述子
	if (HARDNEG) {
		for (int num = 0; num < HardExampleNO && std::getline(finHard, imgName); num++) {
			std::cout << "Now processing hard negative image: " << imgName << std::endl;
			cv::Mat image = cv::imread(imgName);

			std::vector<float> descriptor;
			hog.compute(image, descriptor, cv::Size(8, 8)); // 计算HOG描述子，窗口步长为8*8

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i < DescriptorDim; i++) {
				sampleFeatureMat.at<float>(num + PosSamNO + cropNegNum, i) = descriptor[i];
				sampleLableMat.at<int>(num + PosSamNO + cropNegNum, 0) = -1; // -1: 表示负样本，不含行人
			}
		}
	}

	finNeg.close();
	finPos.close();
	finHard.close();


	// 设置svm
	svm->setType(cv::ml::SVM::C_SVC);
	//CvSVM::C_SVC : C类支持向量分类机。 n类分组  (n≥2)，允许用异常值惩罚因子C进行不完全分类。

	//CvSVM::NU_SVC : 类支持向量分类机。n类似然不完全分类的分类器。
	// 参数为取代C（其值在区间【0，1】中，nu越大，决策边界越平滑）

	//CvSVM::ONE_CLASS : 单分类器，所有的训练数据提取自同一个类里，
	// 然后SVM建立了一个分界线以分割该类在特征空间中所占区域和其它类在特征空间中所占区域。

	//CvSVM::EPS_SVR : 类支持向量回归机。
	// 训练集中的特征向量和拟合出来的超平面的距离需要小于p。异常值惩罚因子C被采用

	//CvSVM::NU_SVR : 类支持向量回归机。 代替了p。

	svm->setC(0.01);// 惩罚因子

	svm->setGamma(1.0);// gamma 参数
	// C 是惩罚系数，即对误差的宽容度。
	// c 越高，说明越不能容忍出现误差,容易过拟合。
	// C 越小，容易欠拟合。C过大或过小，泛化能力变差

	svm->setKernel(cv::ml::SVM::LINEAR);//设置核函数
	// LINEAR：线性核函数；

	// POLY:多项式核函数；
	/// -d用来设置多项式核函数的最高此项次数；默认值是3
	/// -r用来设置核函数中的coef0，也就是公式中的第二个r，默认值是0。
	// 一般选择1-11：1 3 5 7 9 11，也可以选择2,4，6…

	// RBF:径向机核函数【高斯核函数】；
	/// -g用来设置核函数中的gamma参数设置，默认值是1/k（k是类别数）
	//gamma是选择RBF函数作为kernel后，该函数自带的一个参数。
	// 隐含地决定了数据映射到新的特征空间后的分布，
	// gamma越大，支持向量越少，gamma值越小，支持向量越多。
	// 支持向量的个数影响训练与预测的速度。

	// SIGMOID:神经元的非线性作用函数核函数；
	/// -g用来设置核函数中的gamma参数设置，默认值是1/k（k是类别数）
	/// -r用来设置核函数中的coef0，也就是公式中的第二个r，默认值是0

	// PRECOMPUTED：用户自定义核函数

	//SVM的迭代训练过程的中止条件
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 50000, FLT_EPSILON));

	std::cout << "start training ... " << std::endl;

	svm->train(sampleFeatureMat, cv::ml::ROW_SAMPLE, sampleLableMat);
	// svm ->trainAuto(); //svm自动优化参数

	std::cout << "Finishing training..." << std::endl;

	//储存 SVM 分类器
	svm->save(SvmListFile);

	//return 0;
}
