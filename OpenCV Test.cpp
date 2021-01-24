#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>

#define CLASSIFIER_PATH "C:\\Users\\장예위\\Desktop\\숙제\\avatar control\\shape_predictor_68_face_landmarks.dat"
#define IMG_PATH "C:\\Users\\장예위\\Desktop\\숙제\\avatar control\\장예위.bmp"
#define POSE_PATH "C:\\Users\\장예위\\Desktop\\숙제\\avatar control\\poggers.bmp"
#define TRIANGLE_PATH "C:\\Users\\장예위\\Desktop\\숙제\\avatar control\\triangles.txt"
#define VIDEO_PATH "C:\\Users\\장예위\\Desktop\\숙제\\avatar control\\expression test.mp4"
#define VIDEO_OUTPUT_PATH "video.mp4"
#define POINT_N 68
#define OUTER_POINT_N 27
#define FACELINE_POINT_N 17

using namespace dlib;
using namespace std;
using namespace cv;

Mat getFaceMask(Mat img, Point ppt[POINT_N]) {
    Mat pointCircle;
    Mat mask;
    Point pointList[1][OUTER_POINT_N];
    int i;
    int pointN[] = { OUTER_POINT_N };
    printf("getFaceMask\n");
    for (i = 0; i < FACELINE_POINT_N; i++) {
        pointList[0][i] = ppt[i];
    }
    for (i = FACELINE_POINT_N; i < OUTER_POINT_N; i++) {
        pointList[0][i] = ppt[OUTER_POINT_N - (i - FACELINE_POINT_N) - 1];
    }
    const Point* pointArr[1] = { pointList[0] };
    mask = Mat::zeros(img.size(), CV_8UC3);
    pointCircle = Mat::zeros(img.size(), CV_8UC3);
    //imshow("mask", mask);
    waitKey(10);
    /*for (i = 0; i < OUTER_POINT_N; i++) {
        //printf("%d:%d %d\n", i, pointArr[0][i].x, pointArr[0][i].y);
        circle(pointCircle, pointArr[0][i], 10, Scalar(255, 255, 255), -1);
    }*/
    fillPoly(mask, pointArr, pointN, 1, Scalar(255, 255, 255));
   // printf("done fillPoly");
    //imshow("mask", mask);
    //imshow("pointCircle", pointCircle);
    waitKey(10);
    return mask;
}

int findPoint(Point list[POINT_N], int x, int y) {
    int i;
    for (i = 0; i < POINT_N; i++) {
        if (list[i].x == x && list[i].y == y) {
            return i;
        }
    }
    return -1;
}

Mat getTriangleMask(Mat img, Point ppt[3]) {
    Mat pointCircle;
    Mat mask;
    Point pointList[1][OUTER_POINT_N];
    int i;
    int pointN[] = { 3 };
    //printf("getFaceMask\n");
    for (i = 0; i < 3; i++) {
        pointList[0][i] = ppt[i];
    }
    const Point* pointArr[1] = { pointList[0] };
    mask = Mat::zeros(img.size(), CV_8UC3);
    fillPoly(mask, pointArr, pointN, 1, Scalar(255, 255, 255));
    // printf("done fillPoly");
    return mask;
}

Mat morphTriangle(Mat originalTri, Point src[3], Point dst[3], Size resultSize) {
    Mat warpMat;
    Mat result;
    Point2f srcf[3], dstf[3];
    int i;
    result = Mat::zeros(resultSize, CV_8UC3);
    for (i = 0; i < 3; i++) {
        srcf[i] = Point2f((float)src[i].x, (float)src[i].y);
        dstf[i] = Point2f((float)dst[i].x, (float)dst[i].y);
    }
    warpMat = getAffineTransform(srcf, dstf);
    warpAffine(originalTri, result, warpMat, resultSize);
    return result;
}

void putFaceOn(Mat* dst, Mat originalFace, Mat background, Size resultSize, Point originalPList[POINT_N], Point posePList[POINT_N], std::vector<std::array<int, 3>>triangleList) {
    int i, j;
    int triangleN;
    std::array<int, 3> trianglePtArr;
    Mat originalImage;
    Mat resultImage;
    Mat onlyTriangle, triMask;
    Mat warpedTriangle;
    Point srcTri[3];
    Point dstTri[3];
    int tmpX, tmpY;

    originalImage = originalFace;
    triangleN = triangleList.size();

    //background.copyTo(resultImage);
    background = Mat::zeros(resultSize, CV_8UC3);
    for (i = 0; i < triangleN; i++) {
        //printf("i:%d\n", i);
        srcTri[0] = originalPList[triangleList[i][0]];
        srcTri[1] = originalPList[triangleList[i][1]];
        srcTri[2] = originalPList[triangleList[i][2]];
        dstTri[0] = posePList[triangleList[i][0]];
        dstTri[1] = posePList[triangleList[i][1]];
        dstTri[2] = posePList[triangleList[i][2]];

        triMask = getTriangleMask(originalImage, srcTri);
        bitwise_and(originalImage, triMask, onlyTriangle);
        warpedTriangle = morphTriangle(onlyTriangle, srcTri, dstTri, resultSize);

        Mat grayWarpedTriangle, copyMask;
        cvtColor(warpedTriangle, grayWarpedTriangle, COLOR_BGR2GRAY);
        threshold(warpedTriangle, copyMask, 1, 255, THRESH_BINARY);
        copyTo(warpedTriangle, resultImage, copyMask);
    }

    imshow("result", resultImage);
    resultImage.copyTo(*dst);
    return;
}

void getPoints(frontal_face_detector detector, shape_predictor sp, Mat image, Point* result) {
    int i, j;
    array2d<rgb_pixel> img;
    std::vector<dlib::rectangle> dets;
    std::vector<full_object_detection> shapes;
    assign_image(img, cv_image<bgr_pixel>(image));
    dets = detector(img);
    //cout << "Number of faces detected: " << dets.size() << endl;
    for (j = 0; j < dets.size(); ++j)
    {
        full_object_detection shape = sp(img, dets[j]);
        shapes.push_back(shape);

    }

    if (shapes.size() > 0) {
        for (i = 0; i < shapes[0].num_parts(); i++) {
            //circle(image, Point(shapes[0].part(i).x(), shapes[0].part(i).y()), 2, Scalar(0, 0, 255), -1);
            result[i] = Point(shapes[0].part(i).x(), shapes[0].part(i).y());
        }
    }
    return;
}

int main()
{
    int i, j;
    Mat image;
    Mat poseImage;
    Mat triangleImg;
    VideoCapture vidCap;
    VideoWriter vidWrit;
    float videoFPS;
    int videoWidth;
    int videoHeight;
    int videoLength;
    std::vector<full_object_detection> shapes;
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    FILE* fOutTri = fopen("triangles.txt", "w");
    FILE* fOutPnt = fopen("Points.txt", "w");
    Point pListOriginal[POINT_N];
    Point pListPose[POINT_N];
    std::vector<std::array<int, 3>> triangle;
    int triangleN;
    FILE* fTriangle = fopen(TRIANGLE_PATH, "r");
    FILE* fPoints = fopen("C:\\Users\\장예위\\Desktop\\숙제\\avatar control\\imagePoint.txt", "r");
    int tmpX, tmpY;
    Mat imgCopy;

    deserialize(CLASSIFIER_PATH) >> sp;
    // video.open(VIDEO_PATH);
    image = imread(IMG_PATH, IMREAD_ANYCOLOR);
    if (image.empty()) {
        printf("can't open image!\n");
        return -1;
    }
    /*if (image.rows > 800 || image.cols > 800) {
        resize(image, image, Size(image.cols / 2, image.rows / 2));
    }*/
    imshow("originalImage", image);
    vidCap.open(VIDEO_PATH);
    if (!vidCap.isOpened()) {
        printf("can't open video!\n");
        return -1;
    }
    videoFPS = vidCap.get(CAP_PROP_FPS);
    videoWidth = vidCap.get(CAP_PROP_FRAME_WIDTH);
    videoHeight = vidCap.get(CAP_PROP_FRAME_HEIGHT);
    videoLength = vidCap.get(CAP_PROP_FRAME_COUNT);
    vidWrit.open(VIDEO_OUTPUT_PATH, VideoWriter::fourcc('M', 'P', '4', 'V'), videoFPS, Size(videoWidth, videoHeight), true);
    if (!vidWrit.isOpened()) {
        printf("can't write video!\n");
        return -1;
    }
    namedWindow("result");
    printf("imread done!\n");
    waitKey(10);

    fscanf(fTriangle, "%d", &triangleN);
    for (i = 0; i < triangleN; i++) {
        std::array<int, 3> tmpArray;
        fscanf(fTriangle, "%d %d %d", &tmpArray[0], &tmpArray[1], &tmpArray[2]);
        triangle.push_back(tmpArray);
    }


    getPoints(detector, sp, image, pListOriginal);
    /*image.copyTo(imgCopy);
    for (i = 0; i < POINT_N; i++) {
        fscanf(fPoints, "%d %d", &tmpX, &tmpY);
        pListOriginal[i] = Point(tmpX, tmpY);
        circle(imgCopy, pListOriginal[i], 5, Scalar(0, 0, 255),-1);
    }
    imshow("imgCopy", imgCopy);*/
    for (i = 0; i < videoLength; i++) {
        vidCap.read(poseImage);
        imshow("poseImage", poseImage);
        waitKey(1);
        getPoints(detector, sp, poseImage, pListPose);
        printf("got points");
        putFaceOn(&triangleImg, image, poseImage, poseImage.size(), pListOriginal, pListPose, triangle);
        imshow("result", triangleImg);
        waitKey(1);
        printf("%d %% done!\n", i * 100 / videoLength);
        vidWrit.write(triangleImg);
    }
    return 0;
}