package org.example.demo1;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class nhandien {
    public static void main(String[] args) {
        // Load thư viện OpenCV
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Đường dẫn đến thư mục dataset
        String datasetPath = "dataset";
        // Đường dẫn đến file mô hình Haar Cascade
        String cascadePath = "haarcascade_frontalface_default.xml";

        // Khởi tạo CascadeClassifier
        CascadeClassifier faceDetector = new CascadeClassifier(cascadePath);

        // Danh sách đặc trưng và nhãn
        List<MatOfFloat> features = new ArrayList<>();
        List<String> labels = new ArrayList<>();

        // Duyệt qua thư mục dataset để trích xuất đặc trưng
        File datasetDir = new File(datasetPath);
        for (File personDir : datasetDir.listFiles()) {
            if (personDir.isDirectory()) {
                for (File imageFile : personDir.listFiles()) {
                    // Đọc ảnh
                    Mat image = Imgcodecs.imread(imageFile.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);

                    // Phát hiện khuôn mặt
                    MatOfRect faceDetections = new MatOfRect();
                    faceDetector.detectMultiScale(image, faceDetections);

                    // Lấy khuôn mặt đầu tiên (nếu có)
                    if (!faceDetections.empty()) {
                        Rect rect = faceDetections.toArray()[0];
                        Mat face = new Mat(image, rect);
                        Imgproc.resize(face, face, new Size(100, 100)); // Resize về kích thước cố định

                        // Trích xuất đặc trưng LBP
                        MatOfFloat feature = LBPFeatureExtractor.extractLBPFeatures(face);

                        // Thêm đặc trưng và nhãn vào danh sách
                        features.add(feature);
                        labels.add(personDir.getName());
                    }
                }
            }
        }

        // Khởi tạo VideoCapture để đọc từ webcam
        VideoCapture videoCapture = new VideoCapture(0);

        // Kiểm tra xem webcam có mở được không
        if (!videoCapture.isOpened()) {
            System.out.println("Không thể mở webcam.");
            return;
        }

        // Tạo cửa sổ để hiển thị video
        String windowName = "Nhận diện khuôn mặt ";
        org.opencv.highgui.HighGui.namedWindow(windowName, org.opencv.highgui.HighGui.WINDOW_AUTOSIZE);

        // Đọc từng khung hình từ video
        Mat frame = new Mat();
        while (true) {
            // Đọc khung hình tiếp theo
            videoCapture.read(frame);

            // Nếu không đọc được khung hình, thoát khỏi vòng lặp
            if (frame.empty()) {
                System.out.println("Không thể đọc khung hình.");
                break;
            }

            // Chuyển ảnh sang grayscale
            Mat grayFrame = new Mat();
            Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);

            // Phát hiện khuôn mặt trong khung hình
            MatOfRect faceDetections = new MatOfRect();
            faceDetector.detectMultiScale(grayFrame, faceDetections);

            // Duyệt qua các khuôn mặt được phát hiện
            for (Rect rect : faceDetections.toArray()) {
                // Cắt khuôn mặt và resize về kích thước cố định
                Mat face = new Mat(grayFrame, rect);
                Imgproc.resize(face, face, new Size(100, 100));

                // Trích xuất đặc trưng LBP của khuôn mặt
                MatOfFloat testFeature = LBPFeatureExtractor.extractLBPFeatures(face);

                // So sánh đặc trưng với dữ liệu đã có
                String predictedLabel = "Unknown";
                double minDistance = Double.MAX_VALUE;

                for (int i = 0; i < features.size(); i++) {
                    double distance = calculateEuclideanDistance(testFeature, features.get(i));
                    if (distance < minDistance) {
                        minDistance = distance;
                        predictedLabel = labels.get(i);
                    }
                }

                // Tính tỉ lệ so sánh (ví dụ: 100% - (khoảng cách / ngưỡng) * 100)
                double threshold = 1000; // Ngưỡng khoảng cách
                double confidence = Math.max(0, 100 - (minDistance / threshold) * 100);
                confidence = Math.min(100, confidence); // Đảm bảo tỉ lệ không vượt quá 100%

                // Hiển thị kết quả
                String text = predictedLabel + " (" + String.format("%.2f", confidence) + "%)";
                Imgproc.putText(frame, text, new Point(rect.x, rect.y - 10),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 255, 0), 2);
                Imgproc.rectangle(frame, new Point(rect.x, rect.y),
                        new Point(rect.x + rect.width, rect.y + rect.height),
                        new Scalar(0, 255, 0), 2);
            }

            // Hiển thị khung hình đã được xử lý
            org.opencv.highgui.HighGui.imshow(windowName, frame);

            // Thoát khỏi vòng lặp nếu nhấn phím 'q'
            if (org.opencv.highgui.HighGui.waitKey(1) == 'q') {
                break;
            }
        }

        // Giải phóng tài nguyên
        videoCapture.release();
        org.opencv.highgui.HighGui.destroyAllWindows();
    }

    // Hàm tính khoảng cách Euclidean giữa hai đặc trưng
    private static double calculateEuclideanDistance(MatOfFloat feature1, MatOfFloat feature2) {
        double sum = 0;
        for (int i = 0; i < feature1.rows(); i++) {
            double diff = feature1.get(i, 0)[0] - feature2.get(i, 0)[0];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
}