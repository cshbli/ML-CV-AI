# OpenCV Blob Detection with Java/C++

## Java

params should be passed in from an .yaml file.

```
// SimpleBlobDetector create has some issues
    FeatureDetector detector = FeatureDetector.create(FeatureDetector.SIMPLEBLOB);
    try {
        File outputFile = File.createTempFile("SimpleBlobDetector", ".YAML");
        FileUtils.writeToFile(outputFile,
                "%YAML:1.0" // java
                        // parameter
                        // backdoor
                        + "\nthresholdStep: 10.0"
                        + "\nminThreshold: 5"
                        + "\nmaxThreshold: 256"
                        + "\nminRepeatability: 2"
                        + "\nminDistBetweenBlobs: 10"
                        + "\nfilterByColor: 1"
                        + "\nblobColor: 255"
                        + "\nfilterByArea: 1"
                        + "\nminArea: 4"
                        + "\nmaxArea: 484"
                        + "\nfilterByCircularity: 1"
                        + "\nminCircularity: 0.1"
                        + "\nmaxCircularity: 3.4028234663852886e+38"
                        + "\nfilterByInertia: 1"
                        + "\nminInertiaRatio: 0.1"
                        + "\nmaxInertiaRatio: 3.4028234663852886e+38"
                        + "\nfilterByConvexity: 1"
                        + "\nminConvexity: 0.1"
                        + "\nmaxConvexity: 3.4028234663852886e+38"
                        + "\n");
        detector.read(outputFile.getAbsolutePath());
        outputFile.delete();
    } catch (Exception e) {
        e.printStackTrace();
    }

    MatOfKeyPoint keyPointMat = new MatOfKeyPoint();
    detector.detect(imageGray, keyPointMat);
    List<KeyPoint> keyPoints = keyPointMat.toList();
    keyPointMat.release();

    return keyPoints;
```

## C++
```
    cv::SimpleBlobDetector::Params params;
    
    params.minDistBetweenBlobs  = 10;
    params.minThreshold         = 5;
    params.maxThreshold         = 256;
    
    params.filterByArea         = true;
    params.minArea              = 4;
    params.maxArea              = 484;
    
    params.filterByCircularity  = true;
    params.minCircularity       = 0.1;
    
    params.filterByConvexity    = true;
    params.minConvexity         = 0.1;
    
    params.filterByInertia      = true;
    params.minInertiaRatio      = 0.1;
    
    params.filterByColor        = true;
    params.blobColor            = 255;
    
    #if CV_MAJOR_VERSION < 3
        SimpleBlobDetector detector(params);
    #else
        cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
    #endif
    
    detector->detect(image_gray, keyPoints);
```
