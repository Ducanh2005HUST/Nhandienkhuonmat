package org.example.demo1;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.Arrays;

public class check {
    public static void main(String[] args) {
        // Load thư viện OpenCV
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Đường dẫn đến file ảnh
        String imagePath1 = "pic/check1.jpg";
        String imagePath2 = "pic/check2.jpg";

        // Đường dẫn đến file mô hình và cấu hình
        String faceDetectionModelWeights = "res10_300x300_ssd_iter_140000_fp16.caffemodel";
        String faceDetectionModelConfig = "deploy.prototxt";
        String faceEmbeddingModelWeights = "nn4.small2.v1.t7";  // Mô hình trích xuất đặc trưng

        // Đọc ảnh
        Mat img1 = Imgcodecs.imread(imagePath1);
        Mat img2 = Imgcodecs.imread(imagePath2);

        // Kiểm tra xem ảnh có được đọc thành công không
        if (img1.empty() || img2.empty()) {
            throw new RuntimeException("Không thể đọc ảnh. Kiểm tra đường dẫn và định dạng ảnh.");
        }

        // Trích xuất đặc trưng khuôn mặt
        Mat faceEmbedding1 = extractFaceEmbedding(img1, faceDetectionModelWeights, faceDetectionModelConfig, faceEmbeddingModelWeights);
        Mat faceEmbedding2 = extractFaceEmbedding(img2, faceDetectionModelWeights, faceDetectionModelConfig, faceEmbeddingModelWeights);

        // So sánh hai khuôn mặt
        double similarity = compareFaceEmbeddings(faceEmbedding1, faceEmbedding2);
        System.out.println("Độ tương đồng giữa hai khuôn mặt: " + similarity);

        // Ngưỡng để xác định có phải cùng một người hay không
        double threshold = 0.6;
        if (similarity > threshold) {
            System.out.println("Hai khuôn mặt giống nhau.");
        } else {
            System.out.println("Hai khuôn mặt khác nhau.");
        }
    }

    // Hàm trích xuất đặc trưng khuôn mặt
    private static Mat extractFaceEmbedding(Mat image, String faceDetectionModelWeights, String faceDetectionModelConfig, String faceEmbeddingModelWeights) {
        // Load mô hình phát hiện khuôn mặt
        Net faceDetectionNet = Dnn.readNetFromCaffe(faceDetectionModelConfig, faceDetectionModelWeights);
        if (faceDetectionNet.empty()) {
            throw new RuntimeException("Không thể load mô hình phát hiện khuôn mặt. Kiểm tra đường dẫn và file mô hình.");
        }

        // Resize ảnh về kích thước 300x300
        Imgproc.resize(image, image, new Size(300, 300));

        // Chuẩn bị ảnh đầu vào cho mô hình phát hiện khuôn mặt
        Mat blob = Dnn.blobFromImage(image, 1.0, new Size(300, 300), new Scalar(104, 177, 123), false, false);
        faceDetectionNet.setInput(blob);

        // Chạy mô hình để phát hiện khuôn mặt
        Mat detections = faceDetectionNet.forward();

        // Kiểm tra xem detections có rỗng hay không
        if (detections.empty()) {
            throw new RuntimeException("Không tìm thấy khuôn mặt trong ảnh.");
        }

        // In ra kích thước và giá trị của detections để debug
        System.out.println("Kích thước detections: " + detections.size());
        for (int i = 0; i < detections.size().height; i++) {
            double[] detection = detections.get(0, i);
            if (detection == null) {
                System.out.println("Detection " + i + ": null");
            } else {
                System.out.println("Detection " + i + ": " + Arrays.toString(detection));
            }
        }

        // Lấy khuôn mặt đầu tiên
        for (int i = 0; i < detections.size().height; i++) {
            double[] detection = detections.get(0, i);
            if (detection == null) {
                continue;  // Bỏ qua nếu detection là null
            }

            double confidence = detection[2];
            if (confidence > 0.3) {  // Giảm ngưỡng tin cậy để tăng khả năng phát hiện
                // Lấy tọa độ khuôn mặt
                int startX = (int) (detection[3] * image.cols());
                int startY = (int) (detection[4] * image.rows());
                int endX = (int) (detection[5] * image.cols());
                int endY = (int) (detection[6] * image.rows());

                // Kiểm tra tọa độ hợp lệ
                if (startX < 0 || startY < 0 || endX > image.cols() || endY > image.rows()) {
                    System.out.println("Tọa độ khuôn mặt không hợp lệ: (" + startX + ", " + startY + ", " + endX + ", " + endY + ")");
                    continue;  // Bỏ qua nếu tọa độ không hợp lệ
                }

                // Cắt khuôn mặt
                Mat face = new Mat(image, new Rect(startX, startY, endX - startX, endY - startY));

                // Load mô hình trích xuất đặc trưng
                Net faceEmbeddingNet = Dnn.readNetFromTorch(faceEmbeddingModelWeights);
                if (faceEmbeddingNet.empty()) {
                    throw new RuntimeException("Không thể load mô hình trích xuất đặc trưng.");
                }

                // Chuẩn bị ảnh đầu vào cho mô hình trích xuất đặc trưng
                Mat faceBlob = Dnn.blobFromImage(face, 1.0 / 255, new Size(96, 96), new Scalar(0, 0, 0), true, false);
                faceEmbeddingNet.setInput(faceBlob);

                // Trích xuất đặc trưng
                Mat embedding = faceEmbeddingNet.forward();
                return embedding;
            }
        }

        throw new RuntimeException("Không tìm thấy khuôn mặt trong ảnh.");
    }

    // Hàm so sánh hai đặc trưng khuôn mặt
    private static double compareFaceEmbeddings(Mat embedding1, Mat embedding2) {
        // Tính toán cosine similarity giữa hai vector đặc trưng
        double dotProduct = embedding1.dot(embedding2);
        double norm1 = Core.norm(embedding1, Core.NORM_L2);
        double norm2 = Core.norm(embedding2, Core.NORM_L2);
        return dotProduct / (norm1 * norm2);
    }
}