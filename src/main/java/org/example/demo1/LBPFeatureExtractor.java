package org.example.demo1;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.imgproc.Imgproc;

import java.util.List;

public class LBPFeatureExtractor {
    // Hàm tính toán LBP từ ảnh khuôn mặt
    public static MatOfFloat extractLBPFeatures(Mat face) {
        Mat lbpImage = new Mat(face.rows(), face.cols(), CvType.CV_8UC1);
        MatOfFloat features = new MatOfFloat();

        // Chuyển ảnh sang LBP
        for (int i = 1; i < face.rows() - 1; i++) {
            for (int j = 1; j < face.cols() - 1; j++) {
                double center = face.get(i, j)[0];
                StringBuilder binary = new StringBuilder();
                binary.append((face.get(i - 1, j - 1)[0] >= center) ? "1" : "0");
                binary.append((face.get(i - 1, j)[0] >= center) ? "1" : "0");
                binary.append((face.get(i - 1, j + 1)[0] >= center) ? "1" : "0");
                binary.append((face.get(i, j + 1)[0] >= center) ? "1" : "0");
                binary.append((face.get(i + 1, j + 1)[0] >= center) ? "1" : "0");
                binary.append((face.get(i + 1, j)[0] >= center) ? "1" : "0");
                binary.append((face.get(i + 1, j - 1)[0] >= center) ? "1" : "0");
                binary.append((face.get(i, j - 1)[0] >= center) ? "1" : "0");
                int lbpValue = Integer.parseInt(binary.toString(), 2);
                lbpImage.put(i, j, lbpValue);
            }
        }

        // Tính histogram của ảnh LBP
        Imgproc.calcHist(List.of(lbpImage), new MatOfInt(0), new Mat(), features, new MatOfInt(256), new MatOfFloat(0, 256));
        return features;
    }
}