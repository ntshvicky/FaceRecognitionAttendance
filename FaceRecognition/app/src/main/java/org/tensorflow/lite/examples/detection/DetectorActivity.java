/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;

import android.graphics.PointF;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.hardware.camera2.CameraCharacteristics;
import android.media.ImageReader.OnImageAvailableListener;

import android.os.Bundle;
import android.os.SystemClock;
import android.util.Base64;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.face.FaceLandmark;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.examples.detection.WebServices.ConstantString;
import org.tensorflow.lite.examples.detection.WebServices.GetJsonWithParameter;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.SimilarityClassifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  private static final String TAG = "FACE RECOGNITION APP";

  // MobileFaceNet
  private static final int TF_OD_API_INPUT_SIZE = 112;
  private static final boolean TF_OD_API_IS_QUANTIZED = false;
  private static final String TF_OD_API_MODEL_FILE = "mobile_face_net.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
  private static final boolean MAINTAIN_ASPECT = false;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private SimilarityClassifier detector;

  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap portraitBmp = null;
  // here the face is cropped and drawn
  private Bitmap faceBmp = null;

  private boolean computingDetection = false;
  private long timestamp = 0;
  private long lastProcessingTimeMs;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;
  private BorderedText borderedText;

  // Face detector
  private FaceDetector faceDetector;
  float smileProb = 0, rightEyeOpenProb = 0, leftEyeOpenProb = 0;

  /* Set face_detected flag to true when face detected and false when response returned by api */
  private boolean face_detected = false;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    // Real-time contour detection of multiple faces
    FaceDetectorOptions options =
            new FaceDetectorOptions.Builder()
                    .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                    .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                    .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
                    .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                    .build();


    FaceDetector detector = FaceDetection.getClient(options);

    faceDetector = detector;

  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
            TypedValue.applyDimension(
                    TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);
    
    try {
      detector =
              TFLiteObjectDetectionAPIModel.create(
                      getAssets(),
                      TF_OD_API_MODEL_FILE,
                      TF_OD_API_LABELS_FILE,
                      TF_OD_API_INPUT_SIZE,
                      TF_OD_API_IS_QUANTIZED);
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
              Toast.makeText(
                      getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);


    int targetW, targetH;
    if (sensorOrientation == 90 || sensorOrientation == 270) {
      targetH = previewWidth;
      targetW = previewHeight;
    }
    else {
      targetW = previewWidth;
      targetH = previewHeight;
    }
    int cropW = (int) (targetW / 2.0);
    int cropH = (int) (targetH / 2.0);

    croppedBitmap = Bitmap.createBitmap(cropW, cropH, Config.ARGB_8888);

    portraitBmp = Bitmap.createBitmap(targetW, targetH, Config.ARGB_8888);
    faceBmp = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, Config.ARGB_8888);

    frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    cropW, cropH,
                    sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
            canvas -> {
              tracker.draw(canvas);
              if (isDebug()) {
                tracker.drawDebug(canvas);
              }
            });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }


  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;

    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    InputImage image = InputImage.fromBitmap(croppedBitmap, 0);
    faceDetector
            .process(image)
            .addOnSuccessListener(faces -> {
              if (faces.size() == 0) {
                updateResults(currTimestamp, new LinkedList<>());
                return;
              }
              runInBackground(
                      () -> {
                        onFacesDetected(currTimestamp, faces, true);
                        face_detected = false;
                      });
            });


  }

  @Override
  protected int getLayoutId() {
    return R.layout.tfe_od_camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  public void loadAPIResponse(String result, SimilarityClassifier.Recognition rec) {
    Log.w("API Response", result);

    try {
      JSONArray jsonArray = new JSONArray("[" + result + "]");
      JSONObject obj = jsonArray.getJSONObject(0);
      if ("success".equalsIgnoreCase(obj.getString("message"))) {

        JSONArray resArray = obj.getJSONArray("data");
        JSONObject resObj = resArray.getJSONObject(0);
        if(resObj.getString("name").equalsIgnoreCase("unknown")) {
          // If face unknown , then no need to show toast
        } else {
          // If face match or get any result from api show here
          rec.setColor(Color.GREEN);
          rec.setExtra(true);
          rec.setTitle(resObj.getString("name"));
          detector.register(resObj.getString("name"), rec);
          Toast.makeText(getApplicationContext(), obj.getString("data"), Toast.LENGTH_LONG).show();
          //Make face_detect false to prevent looping in toast message
          // Show it will show you result once only
          //Modify as per your need.. I am commenting this line just for output
          face_detected = false;
        }
      }
    }
    catch (Exception e) { }

  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }

  // Face Processing
  private Matrix createTransform(
          final int srcWidth,
          final int srcHeight,
          final int dstWidth,
          final int dstHeight,
          final int applyRotation) {

    Matrix matrix = new Matrix();
    if (applyRotation != 0) {
      if (applyRotation % 90 != 0) {
        LOGGER.w("Rotation of %d % 90 != 0", applyRotation);
      }

      // Translate so center of image is at origin.
      matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

      // Rotate around origin.
      matrix.postRotate(applyRotation);
    }

    if (applyRotation != 0) {
      // Translate back from origin centered reference to destination frame.
      matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
    }

    return matrix;
  }

  private void updateResults(long currTimestamp, final List<SimilarityClassifier.Recognition> mappedRecognitions) {

    tracker.trackResults(mappedRecognitions, currTimestamp);
    trackingOverlay.postInvalidate();
    computingDetection = false;

    Log.w("mappedRecognitions", String.valueOf(mappedRecognitions.size()));

    if (mappedRecognitions.size() > 0) {

      Log.w("face_detected", String.valueOf(face_detected));

      //Post image to api and validate from api
      if(face_detected == true) {
        try {
          ByteArrayOutputStream baos = new ByteArrayOutputStream();
          croppedBitmap.compress(Bitmap.CompressFormat.JPEG, 100, baos);
          byte[] imageBytes = baos.toByteArray();
          String encodedImage = Base64.encodeToString(imageBytes, Base64.DEFAULT);

          JSONObject postDataParams = new JSONObject();
          postDataParams.put("face", encodedImage);
          new GetJsonWithParameter(DetectorActivity.this, ConstantString.GETFACEDETECT_URL, ConstantString.FACEDETECT, postDataParams, mappedRecognitions.get(0)).execute();

        } catch (JSONException jex) {
          Log.w(TAG, "Exception:" + jex.toString());
        }
      }

      LOGGER.i("Adding results");

    }

    runOnUiThread(
            new Runnable() {
              @Override
              public void run() {
                showFrameInfo(previewWidth + "x" + previewHeight);
                showCropInfo(croppedBitmap.getWidth() + "x" + croppedBitmap.getHeight());
                showInference(lastProcessingTimeMs + "ms");
                showSmiling(smileProb);
                showREyeOpened(rightEyeOpenProb);
                showLEyeOpened(leftEyeOpenProb);
              }
            });

  }

  private void onFacesDetected(long currTimestamp, List<Face> faces, boolean face_detected_flag) {

    final List<SimilarityClassifier.Recognition> mappedRecognitions = new LinkedList<>();
    face_detected = face_detected_flag;

    // Note this can be done only once
    int sourceW = rgbFrameBitmap.getWidth();
    int sourceH = rgbFrameBitmap.getHeight();
    int targetW = portraitBmp.getWidth();
    int targetH = portraitBmp.getHeight();
    Matrix transform = createTransform(
            sourceW,
            sourceH,
            targetW,
            targetH,
            sensorOrientation);
    final Canvas cv = new Canvas(portraitBmp);

    // draws the original image in portrait mode.
    cv.drawBitmap(rgbFrameBitmap, transform, null);
    final Canvas cvFace = new Canvas(faceBmp);

    for (Face face : faces) {

      LOGGER.i("FACE " + face.toString());
      LOGGER.i("Running detection on face " + currTimestamp);

      final RectF boundingBox = new RectF(face.getBoundingBox());

      // Read Head movements
      String headangle = "";
      float rotX = face.getHeadEulerAngleX();  // Head is rotated to the right rotY degrees
      headangle += "Head Updown Angle " + rotX;
      float rotY = face.getHeadEulerAngleY();  // Head is rotated to the right rotY degrees
      headangle += "\nHead Side Angle " + rotY;
      float rotZ = face.getHeadEulerAngleZ();  // Head is tilted sideways rotZ degrees
      headangle += "\nHead Tiled Angle " + rotZ;
      LOGGER.i("Head Angle " + headangle);

      /*
        Read face landmark, If landmark detection was enabled (mouth, ears, eyes, cheeks,and nose available)
       */
      FaceLandmark leftEar = face.getLandmark(FaceLandmark.LEFT_EAR);
      if (leftEar != null) {
        PointF leftEarPos = leftEar.getPosition();
        Log.w("Left Ear ", leftEarPos+"");
      }

      // If contour detection was enabled:
      //List<PointF> upperLipBottomContour = face.getContour(FaceContour.UPPER_LIP_BOTTOM).getPoints();

      // If classification was enabled:
      if (face.getSmilingProbability() != null) {
        smileProb = face.getSmilingProbability();
      }
      if (face.getRightEyeOpenProbability() != null) {
        //List<PointF> rightEyeContour = face.getContour(FaceContour.RIGHT_EYE).getPoints();
        rightEyeOpenProb = face.getRightEyeOpenProbability();
      }
      if (face.getLeftEyeOpenProbability() != null) {
        //List<PointF> leftEyeContour = face.getContour(FaceContour.LEFT_EYE).getPoints();
        leftEyeOpenProb = face.getLeftEyeOpenProbability();
      }

      final boolean goodConfidence = true;
      if (boundingBox != null && goodConfidence) {

        // maps crop coordinates to original
        cropToFrameTransform.mapRect(boundingBox);

        // maps original coordinates to portrait coordinates
        RectF faceBB = new RectF(boundingBox);
        transform.mapRect(faceBB);

        // translates portrait to origin and scales to fit input inference size
        float sx = ((float) TF_OD_API_INPUT_SIZE) / faceBB.width();
        float sy = ((float) TF_OD_API_INPUT_SIZE) / faceBB.height();
        Matrix matrix = new Matrix();
        matrix.postTranslate(-faceBB.left, -faceBB.top);
        matrix.postScale(sx, sy);

        cvFace.drawBitmap(portraitBmp, matrix, null);

        String label = "";
        float confidence = -1f;
        Integer color = Color.BLUE;
        Object extra = null;
        Bitmap crop = null;

        if (face_detected_flag) {
          try {
            crop = Bitmap.createBitmap(portraitBmp,
                    (int) faceBB.left,
                    (int) faceBB.top,
                    (int) faceBB.width(),
                    (int) faceBB.height());
          } catch (Exception ex){}
        }


        try {
          final long startTime = SystemClock.uptimeMillis();
          final List<SimilarityClassifier.Recognition> resultsAux = detector.recognizeImage(faceBmp, true);
          lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

          if (resultsAux.size() > 0) {

            SimilarityClassifier.Recognition result = resultsAux.get(0);

            extra = result.getExtra();
            float conf = result.getDistance();
            if (conf < 1.0f) {

              confidence = conf;
              label = result.getTitle();
              if (result.getId().equals("0")) {
                color = Color.GREEN;
              } else {
                color = Color.RED;
              }
            }

          }
        }
        catch (Exception ex){}

        if (getCameraFacing() == CameraCharacteristics.LENS_FACING_FRONT) {
          // camera is frontal so the image is flipped horizontally
          Matrix flip = new Matrix();
          if (sensorOrientation == 90 || sensorOrientation == 270) {
            flip.postScale(1, -1, previewWidth / 2.0f, previewHeight / 2.0f);
          }
          else {
            flip.postScale(-1, 1, previewWidth / 2.0f, previewHeight / 2.0f);
          }
          flip.mapRect(boundingBox);
        }

        final SimilarityClassifier.Recognition result = new SimilarityClassifier.Recognition(
                "0", label, confidence, boundingBox);

        result.setColor(color);
        result.setLocation(boundingBox);
        result.setExtra(extra);
        result.setCrop(crop);
        mappedRecognitions.add(result);
      }
    }

    updateResults(currTimestamp, mappedRecognitions);
  }

  @Override
  public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
    switch (requestCode) {

      case 1:
        Log.w("total Grant", grantResults.length +"");
        if (grantResults.length > 0) {

          boolean CameraAccessPermission = grantResults[0] == PackageManager.PERMISSION_GRANTED;

          if (CameraAccessPermission) {
            Toast.makeText(this, "Permission Granted", Toast.LENGTH_SHORT).show();
          }
          else {
            Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
          }
        }
        break;
    }
  }


}