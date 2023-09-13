# OpenCV

## Change all instances of a colour to a different one in C++/Java

```
Mat mask;
inRange(src, Scalar(255, 255, 255), Scalar(255, 255, 255), mask);
src.setTo(Scalar(0, 0, 0), mask);
```

