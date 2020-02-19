#pragma once

int deal_image() {

	std::ofstream fout;
	// ����ѵ��������
	std::vector<cv::String> NegFileList;
	cv::glob(NEGTRAINFile, NegFileList);
	// ����ѵ��������
	std::vector<cv::String> PosFileList;
	cv::glob(POSTRAINFile, PosFileList);

	// ����data�ļ��к�txt�ļ�
	if (NegFileList.size() == 0 || PosFileList.size() == 0) {
		std::cout << "ѵ���ļ�������" << std::endl;
		return -1;
	}

	if (_access(".//data", 0) != 0) {
		int flag = _mkdir(".//data");
		if (flag != 0) {
			std::cout << "�����ļ�ʧ��" << std::endl;
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
	
	// ���ø�����
	fin.open(NegativeImageList);

	fout.open(".//data//neg.txt", std::ios::trunc);
	// ÿ�Ÿ���������ü���10��
	for (int i = 0; i < cropNegNum && std::getline(fin, ImgName); i++) {

		image = cv::imread(ImgName);
		// �ü�����ͼƬ��С��64*128�����Ա��ü���ͼƬ��С��64*128
		if (image.cols > 64 && image.rows > 128) {
			srand((unsigned)time(NULL));
			for (int j = 0; j < 10; j++) {
				int x = rand() % (image.cols - 64);
				int y = rand() % (image.rows - 128);
				cv::Mat ImageROI = image(cv::Rect(x, y, 64, 128));
				sprintf_s(FileName, 256, ".//img_dir//neg//neg_%06d.png", ++CropNegNumCount);
				cv::imwrite(FileName, ImageROI);
				fout << ".//img_dir//neg//neg_" << std::setw(6) << std::setfill('0') << std::to_string(CropNegNumCount) << ".png" << std::endl;
				std::cout << "�Ѵ����" << CropNegNumCount << "�Ÿ�����" << std::endl;
			}
		}

	}

	std::cout << "һ���ü������������ţ�: " << CropNegNumCount << std::endl;
	CropNegNumCount = 0;

	fin.close();
	fout.close();

	// ����������
	fout.open(".//data//pos.txt", std::ios::trunc);
	fin.open(PostiveImageList);

	for (int i = 0; i < PosSamNO && std::getline(fin, ImgName); i++) {

		image = cv::imread(ImgName);

		if (image.cols > 64 && image.rows > 128) {
			int x = image.cols / 2 - 32; // ���м����
			int y = image.rows / 2 - 64;
			cv::Mat ImageROI = image(cv::Rect(x, y, 64, 128));
			sprintf_s(FileName, 256, ".//img_dir//pos//pos_%06d.png", ++CropNegNumCount);
			cv::imwrite(FileName, ImageROI);
			fout << ".//img_dir//pos//pos_" << std::setw(6) << std::setfill('0') << std::to_string(CropNegNumCount) << ".png" << std::endl;
			std::cout << "�Ѵ����" << CropNegNumCount << "��������" << std::endl;

		}
	}
	std::cout << "һ���ü������������ţ�: " << CropNegNumCount << std::endl;

	fin.close();
	fout.close();

	return 0;
}


// ѵ��svm
void svm_train(cv::HOGDescriptor& hog, int& DescriptorDim, cv::Ptr<cv::ml::SVM>& svm){


	std::string imgName; // ͼƬ��
	std::ifstream finNeg(NegSamListFile); // �������ļ���
	std::ifstream finPos(PosSamListFile); // �������ļ���
	std::ifstream finHard(HardSamListFile); // �����ļ���

	// �ж��ļ��Ƿ����
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

	// ����������������HOG������
	for (int num = 0; num < PosSamNO && std::getline(finPos, imgName); num++) {
		std::cout << "Now processing original positive image: " << imgName << std::endl;
		cv::Mat image = cv::imread(imgName);

		std::vector<float> descriptor;
		hog.compute(image, descriptor, cv::Size(8, 8)); // ����HOG�����ӣ����ڲ���Ϊ8*8

		if (num == 0) {
			DescriptorDim = descriptor.size();
			//std::cout << "DescriptorDim: " << DescriptorDim << std::endl;
			sampleFeatureMat = cv::Mat::zeros(PosSamNO + cropNegNum + HardExampleNO, DescriptorDim, CV_32FC1);
			sampleLableMat = cv::Mat::zeros(PosSamNO + cropNegNum + HardExampleNO, 1, CV_32SC1);
		}

		for (int i = 0; i < DescriptorDim; i++) {
			sampleFeatureMat.at<float>(num, i) = descriptor[i];
			sampleLableMat.at<int>(num, 0) = 1; // 1�� ������
		}
	}
 
	// ��������������HOG������
	for (int num = 0; num < cropNegNum && std::getline(finNeg, imgName); num++) {
		std::cout << "Now processing original negative image: " << imgName << std::endl;
		cv::Mat image = cv::imread(imgName);

		std::vector<float> descriptor;
		hog.compute(image, descriptor, cv::Size(8, 8)); // ����HOG�����ӣ����ڲ���Ϊ8*8

		//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
		for (int i = 0; i < DescriptorDim; i++) {
			sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptor[i];
			sampleLableMat.at<int>(num + PosSamNO, 0) = -1; // -1: ��ʾ����������������
		}
	}

	// ���������� ����HOG������
	if (HARDNEG) {
		for (int num = 0; num < HardExampleNO && std::getline(finHard, imgName); num++) {
			std::cout << "Now processing hard negative image: " << imgName << std::endl;
			cv::Mat image = cv::imread(imgName);

			std::vector<float> descriptor;
			hog.compute(image, descriptor, cv::Size(8, 8)); // ����HOG�����ӣ����ڲ���Ϊ8*8

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int i = 0; i < DescriptorDim; i++) {
				sampleFeatureMat.at<float>(num + PosSamNO + cropNegNum, i) = descriptor[i];
				sampleLableMat.at<int>(num + PosSamNO + cropNegNum, 0) = -1; // -1: ��ʾ����������������
			}
		}
	}

	finNeg.close();
	finPos.close();
	finHard.close();


	// ����svm
	svm->setType(cv::ml::SVM::C_SVC);
	//CvSVM::C_SVC : C��֧������������� n�����  (n��2)���������쳣ֵ�ͷ�����C���в���ȫ���ࡣ

	//CvSVM::NU_SVC : ��֧�������������n����Ȼ����ȫ����ķ�������
	// ����Ϊȡ��C����ֵ�����䡾0��1���У�nuԽ�󣬾��߽߱�Խƽ����

	//CvSVM::ONE_CLASS : �������������е�ѵ��������ȡ��ͬһ�����
	// Ȼ��SVM������һ���ֽ����Էָ�����������ռ�����ռ������������������ռ�����ռ����

	//CvSVM::EPS_SVR : ��֧�������ع����
	// ѵ�����е�������������ϳ����ĳ�ƽ��ľ�����ҪС��p���쳣ֵ�ͷ�����C������

	//CvSVM::NU_SVR : ��֧�������ع���� ������p��

	svm->setC(0.01);// �ͷ�����

	svm->setGamma(1.0);// gamma ����
	// C �ǳͷ�ϵ�����������Ŀ��ݶȡ�
	// c Խ�ߣ�˵��Խ�������̳������,���׹���ϡ�
	// C ԽС������Ƿ��ϡ�C������С�������������

	svm->setKernel(cv::ml::SVM::LINEAR);//���ú˺���
	// LINEAR�����Ժ˺�����

	// POLY:����ʽ�˺�����
	/// -d�������ö���ʽ�˺�������ߴ��������Ĭ��ֵ��3
	/// -r�������ú˺����е�coef0��Ҳ���ǹ�ʽ�еĵڶ���r��Ĭ��ֵ��0��
	// һ��ѡ��1-11��1 3 5 7 9 11��Ҳ����ѡ��2,4��6��

	// RBF:������˺�������˹�˺�������
	/// -g�������ú˺����е�gamma�������ã�Ĭ��ֵ��1/k��k���������
	//gamma��ѡ��RBF������Ϊkernel�󣬸ú����Դ���һ��������
	// �����ؾ���������ӳ�䵽�µ������ռ��ķֲ���
	// gammaԽ��֧������Խ�٣�gammaֵԽС��֧������Խ�ࡣ
	// ֧�������ĸ���Ӱ��ѵ����Ԥ����ٶȡ�

	// SIGMOID:��Ԫ�ķ��������ú����˺�����
	/// -g�������ú˺����е�gamma�������ã�Ĭ��ֵ��1/k��k���������
	/// -r�������ú˺����е�coef0��Ҳ���ǹ�ʽ�еĵڶ���r��Ĭ��ֵ��0

	// PRECOMPUTED���û��Զ���˺���

	//SVM�ĵ���ѵ�����̵���ֹ����
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 50000, FLT_EPSILON));

	std::cout << "start training ... " << std::endl;

	svm->train(sampleFeatureMat, cv::ml::ROW_SAMPLE, sampleLableMat);
	// svm ->trainAuto(); //svm�Զ��Ż�����

	std::cout << "Finishing training..." << std::endl;

	//���� SVM ������
	svm->save(SvmListFile);

	//return 0;
}
