#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <cstdlib>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat& frame, int nr);
/** Global variables */
const cv::String	WINDOW_NAME("Camera video");
/** place the xml in the home directory */
const cv::String	CASCADE_FILE("/home/haar_xml_07_19.xml");
cv::CascadeClassifier *cascade_classifier;
cv::Mat frame, traffic_template;
cv::Mat trLightROI;

void showFrame(cv::Mat &frame)
{
    cv::imshow(WINDOW_NAME, frame);
    int c = waitKey(50);
    if ((char)c == 30) { exit(0);}
}
/** Circle Detection after detecting a traffic light */
void DetectCircles(cv::Mat &traffic_template, int nr){
	std::vector<cv::Vec3f> VectorCir;
	std::vector<cv::Vec3f>::iterator iterCircles;
	Mat resultImg;

	//Save mat of detected circle in traffic lights
	string circle_name = "circle";
	string type_circle = ".jpg";
	string folderNameCircle = "DetectedCircles";
	stringstream ssfn_circle;
	ssfn_circle << folderNameCircle << "/" << circle_name << (nr) << type_circle;

	string fullpath_circle = ssfn_circle.str();
	ssfn_circle.str("");

	//Apply color map to search for certian color
	applyColorMap(traffic_template,traffic_template,COLORMAP_SUMMER);
	cv::inRange(traffic_template,cv::Scalar(0,90,90),cv::Scalar(204,255,255),resultImg);
	cv::GaussianBlur(resultImg,resultImg,cv::Size(9,9),0.5,0.5);
	cv::HoughCircles(resultImg,
				 VectorCir,
				 CV_HOUGH_GRADIENT,
				 2,
				 90,
				 50,
				 20,
				 4,
				 10);
	
	for(iterCircles = VectorCir.begin(); iterCircles !=VectorCir.end(); iterCircles++) {

	cv::circle(traffic_template,
		cv::Point((int)(*iterCircles)[0],(int)(*iterCircles)[1]),
		3,
		cv::Scalar(255,0,0),
		CV_FILLED);

	cv::circle(traffic_template,cv::Point((int)(*iterCircles)[0],(int)(*iterCircles)[1]),
		(int)(*iterCircles)[2],
		cv::Scalar(0,0,255),
		3);
		
	imwrite(fullpath_circle,trLightROI);
	}
	// showing the the process of detecting circles
	//imshow("processed",resultImg);
 }


/** Detect trafficlights */
void detectAndDisplay(Mat& frame, int nr){
	
	std::vector<cv::Rect> trLights;
	Mat haar_detection;
	string haar_name = "haar";
	string type_haar = ".jpg";
	string folderNameHaar = "DetectedHAAR";
	stringstream ssfn_haar;
	ssfn_haar <<folderNameHaar << "/" << haar_name << (nr) << type_haar;
	string fullpath_haar = ssfn_haar.str();
	ssfn_haar.str("");
		
	//-- Detect TrafficLights through cascade classifier
	cascade_classifier->detectMultiScale(frame, trLights, 1.1, 0, CV_HAAR_SCALE_IMAGE | CV_HAAR_FEATURE_MAX, Size(24, 24));
	frame.copyTo(traffic_template);
	for (size_t i = 0; i < trLights.size(); i++){
		Rect trLights_i =  trLights[i];
		haar_detection = frame(trLights_i);
		trLightROI = traffic_template(trLights[i]);
		imwrite(fullpath_haar,haar_detection);
	}
	waitKey(10);
	if(trLights.empty()){
		return;
	}
	DetectCircles(trLightROI, nr);
	//Use if you want to see the frame
	//imshow("show", trLightROI);
}

int main(void){

	//Create folders for the detected trafficlights 
	const int dir_Circle = mkdir("DetectedCircles", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	const int dir_Haar = mkdir("DetectedHAAR", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	//Capture function for the camera 0 or 1 works
	VideoCapture capture(0);	
	// Calling the xml trained haar	
	cascade_classifier = new cv::CascadeClassifier(CASCADE_FILE);
	// counter for the saved MATs	
	int nr = 0;
	if(!capture.isOpened()){ 
		printf("--(!)Error opening video capture\n"); 
	return -1; 
	}
    	if (cascade_classifier->empty()){
        std::cout << "Error creating cascade classifier. Make sure the file \n"
            "\t" << CASCADE_FILE << "\n"
            "is in working directory.\n";
        exit(1);
    	}
	while (true){	
		//while the frames are running start the detection		
		capture >> frame;
		detectAndDisplay(frame,nr);
		//Show the original frame
		//showFrame(frame);
		nr++;
	}
	delete cascade_classifier;
	return 0;
}
