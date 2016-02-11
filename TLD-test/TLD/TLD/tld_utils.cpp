#include "tld_utils.h"		//尖括号--到系统目录找文件，没有则报错
							//双引号--到指定路径找文件，没有给出，则到文件目录查找
using namespace cv;
using namespace std;

void drawBOx(Mat& image, CvRect box, Scalar color, int thick){
	rectangle(image, cvPoint(box.x, box.y), cvPoint(box.x+box.width,box.y+box.height),color, thick);
}

void drawPoints(Mat& image, vector<Point2f> points, Scalar color){
	for(vector<Point2f>::const_iterator i = points.begin(), ie = points.end(); i != ie; ++i)
	{
		Point center( cvRound(i->x), cvRound(i->y));
		circle(image, *i, 2, color, 1);
	}
}
	
Mat createMask(const Mat& image, CvRect box){
	Mat mask = Mat::zeros(image.rows, image.cols, CV_8U);
	drawBox(mask,box, Scalar::all(255),CV_FILLED);
	return mask;
}

float median(vector<float> v)
{
	int n = floor(v.size() / 2.0f);
	nth_element(v.begin(), v.begin()+n, v.end());
	return v[n];
}
		
vector<int> index_shuffle(int begin, int end){
	vector<int> indexes(end- begin);
	for(int i=begin; i<end; i++){
		indexes[i]=i;
	}
	random_shuffle(indexes.begin(),indexes.end());
	return indexes;
}
				