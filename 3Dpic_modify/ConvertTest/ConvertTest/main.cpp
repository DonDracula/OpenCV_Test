#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include "opencv2/imgproc/imgproc.hpp"  
#include <fstream>
#include <windows.h>  

#define SCALE 1.7f
#define CHART_SIZE 470.0*SCALE			//size of the modified chart
#define CROSS_LENGTH 20			//length of the cross mark
#define PIXEL_TO_64_COE ((64.0/CHART_SIZE))
#define REAL_LENGTH 42 

double mx = 0, my = 0;
int dx = 0, dy = 0, horizBar_x = 0, vertiBar_y = 0;
bool clickVertiBar = false, clickHorizBar = false, needScroll = false;
CvRect rect_bar_horiz, rect_bar_verti;

using namespace cv;
using namespace std;
const char* wndname = "Vertex Correction Demo";

void mouseHandler(int event, int x, int y, int flags, void *param);
int static times; //记录调用次数  

int selctPointX, selctPointY;
int colsBegin, colsEnd;

void createGlobalCoordinate(Mat src);
void createOriginChart(Mat src);
Mat modified_chart0;
Mat m_image;

//select image point
Mat srcImg, selectedImg, temp;
void releaseImg(Mat a, int x, int y);
void showImage();
Mat findImg(int x, int y);
int m_chart_size;

void mouse_callback(int event, int x, int y, int flags, void* param);
void myShowImageScroll(char* title, IplImage src_img,
	int winWidth, int winHeight); // 显示窗口大小默认为 1400×700  

Point2f convert(vector<Point2f> standChart, vector<Point2f> selectedChart, Point2f inputPoint);

vector<Point2f> pointCoordinateList;
vector<Point2f> standardChartList;
vector<Point2f> originChartList;		//65*65表上各点位置
//vector<Point> currentChartList;
vector<Point2f> outputChartList;
vector<Point2f> deviationList;

int point = -1;			//currently selected point
Mat test(CHART_SIZE, CHART_SIZE, CV_8UC1);
Mat src_img;
int point_chart_width=5;
//int chart_width_0=65;
int chart_width_0 = 5; 
int centerX=0, centerY=0;

std::vector<Point2f> calibrateList;
std::vector<Point2f>m_data, data1, data2, outputData;

void readTxtFile(std::vector<Point2f> data, string filename) {
	char buffer[256];
	std::ifstream myfile(filename);

	Point2f point;
	if (!myfile) {
		std::cout << "Unable to open myfile";
		exit(1); // terminate with error
	}
	while (myfile >> point.x >> point.y)
	{
		std::cout << point.x<< " " << point.y << endl;
		data.push_back(point);
	}
	cout << "  my data size::::::::::::::::::" << data.size() << endl;
	myfile.close();
}

int main(int argc, char** argv) {
	//chart size
	// create window   
	int width = 1400, height = 700;
	src_img = imread("pic6.jpg", CV_LOAD_IMAGE_COLOR);

	//pyrUp(src_img, src_img, Size(src_img.cols * SCALE, src_img.rows * SCALE));
	resize(src_img, src_img, Size(src_img.cols * SCALE, src_img.rows * SCALE),0,0);
	namedWindow("Drag & Drop", WINDOW_AUTOSIZE);
//	namedWindow("Enlarge", CV_WINDOW_AUTOSIZE);

	createGlobalCoordinate(src_img);
	createOriginChart(src_img);
	//滑动条
	//cvSetMouseCallback("Drag & Drop", mouse_callback);
	cvSetMouseCallback("Drag & Drop", mouseHandler, NULL);

	//得到屏幕像素大小，pixel
	float nScreenWidth_pixel, nScreenHeight_pixel;
	nScreenWidth_pixel = GetSystemMetrics(SM_CXSCREEN);
	nScreenHeight_pixel = GetSystemMetrics(SM_CYSCREEN);
	//得到屏幕物理大小，单位mm
	float nScreenWidth_mm, nScreenHeight_mm;
	HDC hdcScreen = GetDC(NULL);   //获取屏幕的HDC  
	nScreenWidth_mm = GetDeviceCaps(hdcScreen, HORZSIZE);
	nScreenHeight_mm = GetDeviceCaps(hdcScreen, VERTSIZE);
	//计算分辨率与屏幕尺寸比例
	float rate_pixel_to_mm = (float)nScreenWidth_mm / nScreenWidth_pixel;

	float ratio = 500 / REAL_LENGTH;
	m_chart_size= 40 * ratio;
	//cout << "width" << m_chart_size << endl;

	//cout << "ddddddddddd"<<ratio << endl;

	imshow("Drag & Drop", src_img);

	showImage();
	//waitKey(0);

	//int width = 1400, height = 800;
	ofstream outputFile("outFile.txt"); //创建待写入数据文件
	//轮询获取按键信息  

//	readTxtFile(data1, "input01.txt");
	char buffer[256];
	std::ifstream myfile01("input01.txt");
	Point2f point01;
	if (!myfile01) {
		std::cout << "Unable to open myfile";
		exit(1); // terminate with error
	}
	while (myfile01 >> point01.x >> point01.y)
	{
		//std::cout << point01.x << " " << point01.y << endl;
		data1.push_back(point01);
	}
	cout << "  my data 1 size::::::::::::::::::" << data1.size() << endl;
	myfile01.close();

	//readTxtFile(data2, "input02.txt");
	std::ifstream myfile02("input02.txt");
	Point2f point02;
	if (!myfile02) {
		std::cout << "Unable to open myfile";
		exit(1); // terminate with error
	}
	while (myfile02 >> point02.x >> point02.y)
	{
	//	std::cout << point02.x << " " << point02.y << endl;
		data2.push_back(point02);
	}
	cout << "  my data 2 size::::::::::::::::::" << data2.size() << endl;
	myfile02.close();

	ofstream finalOutputFile("finalOutputFile.txt"); //创建待写入数据文件
	int key = 0;
	while (1)
	{
		key = waitKey(9);//读取键值到key变量中  
		//根据key变量的值，进行不同的操作  
		switch (key)
		{
		case '5':

			cout << "data1  size: " << data1.size() << endl;
			cout << "data2 size: " << data2.size() << endl;
			for (int i = 0; i < 25; i++)
			{
				Point2f m_point;
				m_point.x = data1[i].x + data2[i].x;
				m_point.y = data1[i].y + data2[i].y;
				m_data.push_back(m_point);
			}


			for (int i = 0; i < m_data.size(); i++)
			{
				finalOutputFile << m_data[i].x << " " << m_data[i].y << endl;//输出打印数组
			}
			finalOutputFile.close();

			break;
		case '0':
			for (int i = 0; i < deviationList.size(); i++)
			{
				outputFile << deviationList[i].x << " " << deviationList[i].y << endl;//输出打印数组
			}
			outputFile.close();
			break;

		case '1'://按键1按下，移动后点击确认，调用resize函数，计算移动距离
		{
			for (int i = 0; i < originChartList.size(); i++)
			{
				Point2f Q;
				Q = convert(standardChartList, pointCoordinateList, originChartList[i]);

				outputChartList.push_back(Q);
				Point2f deviation;

				deviation.x = rate_pixel_to_mm*(originChartList[i].x - Q.x) / SCALE;
				deviation.y = rate_pixel_to_mm*(Q.y - originChartList[i].y) / SCALE;
				//deviation.x = 0;
				//deviation.y = 0;
				deviationList.push_back(deviation);
			}
			cout << "Create Over!!" << endl;
		}
		break;
		case '2':
			//	myShowImageScroll("Drag & Drop", &IplImage(src_img), width, height);
			break;

		case '8':
			Mat tmpImage, dstImage;//临时变量和目标图的定义  
			tmpImage = src_img;//将原始图赋给临时变量  
			resize(tmpImage, src_img, Size(tmpImage.cols * 1.5, tmpImage.rows * 1.5), (0, 0), (0, 0), 3);

			break;	
		}
	}
	//OutFile.close();
	cvDestroyWindow("Drag & Drop");
	src_img.release();

	return 0;
}
/*!
 * create chart
 * @param[in] src: source image
 * @param[in] chart_sizeX: width of chart
 * @param[in] chart_sizeY: height of chart
 * @param[in] flag: if true->create modified chart 0, if false-> create points on the source image
 */
//创建65*65表格（屏幕坐标）
void createOriginChart(Mat src)
{
	for (int i = 0; i < chart_width_0; i++)
	{
		for (int j = 0; j < chart_width_0; j++)
		{
			Point2f p1, p2;
			p1.x = (j*(CHART_SIZE) / (chart_width_0 - 1) - (CHART_SIZE) / 2) + src.size().width / 2,
			p1.y= (i*(CHART_SIZE) / (chart_width_0 - 1) - (CHART_SIZE) / 2) + src.size().height / 2;
			//p1.x = (j*(src.size().width ) / (chart_width_0 - 1) - (src.size().width ) / 2),
			//p1.y= (i*(src.size().width ) / (chart_width_0 - 1) - (src.size().width ) / 2) ;

			originChartList.push_back(p1);
		}
	}
}
//n*n标准表格，及变形后表格
void createGlobalCoordinate(Mat src)
{
	for (int i = 0; i < point_chart_width; i++)
	{
		for (int j = 0; j < point_chart_width; j++)
		{
			Point2f p1, p2;
			p2.x = (j*(CHART_SIZE) / (point_chart_width - 1) - (CHART_SIZE) /2) + src.size().width/2,
				p2.y = (i*(CHART_SIZE) / (point_chart_width - 1) - (CHART_SIZE) / 2) + src.size().height/2;
			//p2.x = (j*(src.size().width ) / (point_chart_width - 1) - (src.size().width ) / 2),
			//	p2.y = (i*(src.size().width ) / (point_chart_width - 1) - (src.size().width ) / 2);
			pointCoordinateList.push_back(p2);
			standardChartList.push_back(p2);  //固定点位置，用于计算最终移动距离
		}
	}
}

//*******************************************************************//  
//鼠标回调函数  
void mouseHandler(int event, int x, int y, int flags, void *param)
{
	Mat m_tmp;

	switch (event) {
	case CV_EVENT_LBUTTONDOWN:		//left button press
		selectedImg = findImg(x, y);
		//m_image = src_img(Range(x - 100, x + 100), Range(y - 100, y + 100));
		//centerX = x, centerY = y;
		//resize(m_image, m_image, Size(m_image.cols * 2, m_image.rows * 2), 0, 0);
		//imshow("Enlarge", m_image);
		break;

	case CV_EVENT_LBUTTONUP:	//left mouse button release
		if ((!selectedImg.empty()) && point != -1) {
			releaseImg(selectedImg, x, y);
			selectedImg.release();
		}
		break;

	case CV_EVENT_MOUSEMOVE:
	{
		/* draw a rectangle*/
		if (point != -1) {
			if (!selectedImg.empty()) {
				temp = selectedImg.clone();
				m_tmp = m_image.clone();
				rectangle(temp,
					Point2f(x - 1, y - 1),
					Point2f(x + 1, y + 1),
					Scalar(0, 0, 255, 0), 2, 8, 0);

				//adjust the lines
				for (int i = 0; i < pointCoordinateList.size(); i++) {
					if (i != point) {
						line(temp,
							Point2f(x, y),
							Point2f(pointCoordinateList[i].x, pointCoordinateList[i].y),
							Scalar(0, 0, 255, 0), 1, 8, 0);
					}
				}
				//rectangle(m_tmp,
				//	Point2f((x - 1)*200/temp.size().width, (y - 1) * 200 / temp.size().width),
				//	Point2f((x + 1) * 200 / temp.size().width, (y + 1) * 200 / temp.size().width),
				//	Scalar(0, 0, 255, 0), 2, 8, 0);

				//imshow("Enlarge", m_tmp);
				imshow("Drag & Drop", temp);
			}
		}
		break;
		
	}

	}
}

void releaseImg(Mat a, int x, int y)
{
	pointCoordinateList[point].x = x;
	pointCoordinateList[point].y = y;
	centerX = x, centerY = y;
	showImage();
}
void showImage()
{
	cout << " center x" << centerX << "center y " << centerY << endl;
	Mat img2 = src_img.clone();

	//draw the points
	for (int j = 0; j<pointCoordinateList.size(); j++) {
		line(img2, Point2f(pointCoordinateList[j].x - 1 - CROSS_LENGTH / 2, pointCoordinateList[j].y - 1), Point2f(pointCoordinateList[j].x - 1 + CROSS_LENGTH / 2, pointCoordinateList[j].y - 1), Scalar(255, 100, 0), 1, 8, 0);
		line(img2, Point2f(pointCoordinateList[j].x - 1, pointCoordinateList[j].y - 1 - CROSS_LENGTH / 2), Point2f(pointCoordinateList[j].x - 1, pointCoordinateList[j].y - 1 + CROSS_LENGTH / 2), Scalar(255, 100, 0), 1, 8, 0);
	}
	imshow("Drag & Drop", img2);
//	Mat img3 = img2(Range(100 - 100, 100 + 100), Range(100 - 100, 100 + 100));
	//imshow("Enlarge", img3);

	img2.release();
//	img3.release();
}

/* 转换函数（双线性插值）
 @ standChart: 输入标准参照表
 @ selectedGrid: 变形后图形表
 @ inputPoint: 输入任意点位置
*/
Point2f convert(vector<Point2f> standChart, vector<Point2f> selectedChart,Point2f inputPoint)
{
	Point2f Q;	//输出点
	float q1, q2, q3, q4, base;
	int selecetedPoint =0;
	//特殊点判断，得到方格左上角编号
	for (int i = 0; i < standChart.size(); i++)
	{
		if (inputPoint.x == standChart[point_chart_width - 1].x&&inputPoint.y >= standChart[i].y&&inputPoint.y < standChart[i + point_chart_width].y)
		{
			selecetedPoint = point_chart_width - 2 + i;
			break;
		}
		else if (inputPoint.y == standChart[standChart.size() - 1].y&&inputPoint.x >= standChart[i].x&&inputPoint.x < standChart[i + 1].x)
		{
			selecetedPoint = i + point_chart_width*(point_chart_width - 2);
			break;
		}
		else if (inputPoint == standChart[standChart.size() - 1])
		{
			selecetedPoint = standChart.size() - 2 - point_chart_width;
			break;
		}
		else if (inputPoint.x >= standChart[i].x && inputPoint.x < standChart[i + 1].x && inputPoint.y >= standChart[i].y&&inputPoint.y < standChart[i + point_chart_width].y)
		{
			selecetedPoint = i;
			break;
		}
	}

	//int posX = (int)((inputPoint.x- src_img.size().width / 2 + CHART_SIZE/2) / (CHART_SIZE / 64));
	//int posY = (int)((CHART_SIZE / 2 - inputPoint.y +src_img.size().width / 2) / (CHART_SIZE /64));
	//posX = max(0, min(63, posX));
	//posY = max(0, min(63, posY));

	//selecetedPoint = posY * 64 + posX;

	//cout << "Input Point==" << inputPoint << "selected  Point" << selecetedPoint << endl;
	//双线性插值转换
	base = (standChart[selecetedPoint + 1].x - standChart[selecetedPoint].x)*(standChart[selecetedPoint + point_chart_width].y - standChart[selecetedPoint].y);
	if (base == 0) { return false; }
	q1 = (standChart[selecetedPoint + 1].x - inputPoint.x)*(standChart[selecetedPoint+ point_chart_width].y- inputPoint.y);
	q2 = (inputPoint.x - standChart[selecetedPoint].x)*( standChart[selecetedPoint+ point_chart_width].y- inputPoint.y);
	q3 = (standChart[selecetedPoint + 1].x - inputPoint.x)*(inputPoint.y-standChart[selecetedPoint].y);
	q4 = (inputPoint.x - standChart[selecetedPoint].x)*(inputPoint.y-standChart[selecetedPoint].y);

	Q = q1*selectedChart[selecetedPoint] / base + q2*selectedChart[selecetedPoint + 1] / base
		+ q3*selectedChart[selecetedPoint + point_chart_width] / base + q4*selectedChart[selecetedPoint + 1 + point_chart_width] / base;
	return Q;
}
//鼠标点击点查找
Mat findImg(int x, int y) 
{
	Mat img = src_img.clone();

	for (int i = 0; i<pointCoordinateList.size(); i++) {
		if ((x >= (pointCoordinateList[i].x - CROSS_LENGTH/5)) && (x <= (pointCoordinateList[i].x + CROSS_LENGTH/5)) && (y <= (pointCoordinateList[i].y + CROSS_LENGTH/5)) && (y <= (pointCoordinateList[i].y + CROSS_LENGTH/5))) {
			point = i;
			break;
		}
	}


	//draw cross
	for (int j = 0; j<pointCoordinateList.size(); j++) {
		if (j != point) {
			line(img, Point2f(pointCoordinateList[j].x - 1 - CROSS_LENGTH / 5, pointCoordinateList[j].y - 1), Point2f(pointCoordinateList[j].x - 1 + CROSS_LENGTH / 5, pointCoordinateList[j].y - 1), Scalar(0, 0, 255), 1, 8, 0);
			line(img, Point2f(pointCoordinateList[j].x - 1, pointCoordinateList[j].y - 1 - CROSS_LENGTH / 5), Point2f(pointCoordinateList[j].x - 1, pointCoordinateList[j].y - 1 + CROSS_LENGTH / 5), Scalar(0, 0, 255), 1, 8, 0);
		}
	}

	return img;
}

void mouse_callback(int event, int x, int y, int flags, void* param)
{
	if (needScroll)
	{
		switch (event)
		{
		case CV_EVENT_LBUTTONDOWN:
			mx = x, my = y;
			dx = 0, dy = 0;
			// 按下左键时光标定位在水平滚动条区域内  
			if (x >= rect_bar_horiz.x && x <= rect_bar_horiz.x + rect_bar_horiz.width
				&& y >= rect_bar_horiz.y && y <= rect_bar_horiz.y + rect_bar_horiz.height)
			{
				clickHorizBar = true;
			}
			// 按下左键时光标定位在垂直滚动条区域内  
			if (x >= rect_bar_verti.x && x <= rect_bar_verti.x + rect_bar_verti.width
				&& y >= rect_bar_verti.y && y <= rect_bar_verti.y + rect_bar_verti.height)
			{
				clickVertiBar = true;
			}
			break;
		case CV_EVENT_MOUSEMOVE:
			if (clickHorizBar)
			{
				dx = fabs(x - mx) > 1 ? (int)(x - mx) : 0;
				dy = 0;
			}
			if (clickVertiBar)
			{
				dx = 0;
				dy = fabs(y - my) > 1 ? (int)(y - my) : 0;
			}
			mx = x, my = y;
			break;
		case CV_EVENT_LBUTTONUP:
			mx = x, my = y;
			dx = 0, dy = 0;
			clickHorizBar = false;
			clickVertiBar = false;
			break;
		default:
			dx = 0, dy = 0;
			break;
		}
	}
}

void myShowImageScroll(char* title, IplImage src_img,
	int winWidth = 1400, int winHeight = 700) // 显示窗口大小默认为 1400×700  
{
	//IplImage* src_img = cvCloneImage(m_srcImage);
	IplImage* dst_img;
	CvRect  rect_dst,   // 窗口中有效的图像显示区域  
		rect_src;   // 窗口图像对应于源图像中的区域  
	int imgWidth = src_img.width,
		imgHeight = src_img.height,
		barWidth = 25;  // 滚动条的宽度（像素）  
	double  scale_w = (double)imgWidth / (double)winWidth,    // 源图像与窗口的宽度比值  用以判断是否超出显示范围
		scale_h = (double)imgHeight / (double)winHeight;            // 源图像与窗口的高度比值  用以判断是否超出显示范围

	if (scale_w < 1)                                                                     //如果小于1 说明原图比窗口小，窗口的宽度将重新赋值
		winWidth = imgWidth + barWidth;
	if (scale_h < 1)                                                                     //如果小于1 说明原图比窗口小，窗口的高度将重新赋值
		winHeight = imgHeight + barWidth;

	int showWidth = winWidth, showHeight = winHeight; // 窗口中有效的图像显示区域的宽和高  
	int src_x = 0, src_y = 0;                                                      // 源图像中 rect_src 的左上角位置  
	int horizBar_width = 0, horizBar_height = 0,                   //定义并初始化垂直于水平滑块的宽高
		vertiBar_width = 0, vertiBar_height = 0;

	needScroll = scale_w > 1.0 || scale_h > 1.0 ? true : false;
	// 若图像大于设定的窗口大小，则显示滚动条  
	if (needScroll)
	{
		dst_img = cvCreateImage(cvSize(winWidth, winHeight), src_img.depth, src_img.nChannels);
		cvZero(dst_img);
		// 源图像宽度大于窗口宽度，则显示水平滚动条  
		if (scale_w > 1.0)  //宽度超出了
		{
			showHeight = winHeight - barWidth;
			horizBar_width = (int)((double)winWidth / scale_w);
			horizBar_height = winHeight - showHeight;
			horizBar_x = min(
				max(0, horizBar_x + dx),
				winWidth - horizBar_width);
			rect_bar_horiz = cvRect(
				horizBar_x,
				showHeight + 1,
				horizBar_width,
				horizBar_height);
			// 显示水平滚动条  
			cvRectangleR(dst_img, rect_bar_horiz, cvScalarAll(255), -1);
		}
		// 源图像高度大于窗口高度，则显示垂直滚动条  
		if (scale_h > 1.0)  //高度超出了
		{
			showWidth = winWidth - barWidth;
			vertiBar_width = winWidth - showWidth;
			vertiBar_height = (int)((double)winHeight / scale_h);
			vertiBar_y = min(
				max(0, vertiBar_y + dy),
				winHeight - vertiBar_height);
			rect_bar_verti = cvRect(
				showWidth + 1,
				vertiBar_y,
				vertiBar_width,
				vertiBar_height);   //确定垂直滚动条的白色部分的大小
									// 显示垂直滚动条  
			cvRectangleR(dst_img, rect_bar_verti, cvScalarAll(255), -1);
		}

		showWidth = min(showWidth, imgWidth);
		showHeight = min(showHeight, imgHeight);
		// 设置窗口显示区的 ROI  
		rect_dst = cvRect(0, 0, showWidth, showHeight);
		cvSetImageROI(dst_img, rect_dst);
		// 设置源图像的 ROI  
		src_x = (int)((double)horizBar_x*scale_w);
		src_y = (int)((double)vertiBar_y*scale_h);
		src_x = min(src_x, imgWidth - showWidth);
		src_y = min(src_y, imgHeight - showHeight);
		rect_src = cvRect(src_x, src_y, showWidth, showHeight);
		cvSetImageROI(&src_img, rect_src);
		// 将源图像内容复制到窗口显示区  
		cvCopy(&src_img, dst_img);
		//dst_img=cvCloneImage(src_img);

		cvResetImageROI(dst_img);
		cvResetImageROI(&src_img);
		// 显示图像和滚动条  
		cvShowImage("Drag & Drop", dst_img);

		cvReleaseImage(&dst_img);
	}
	// 源图像小于设定窗口，则直接显示图像，无滚动条  
	else
	{
		cvShowImage("Drag & Drop", &src_img);
	}
}