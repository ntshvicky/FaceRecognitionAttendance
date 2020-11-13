package org.tensorflow.lite.examples.detection.WebServices;

import android.os.AsyncTask;
import android.util.Log;

import org.json.JSONObject;
import org.tensorflow.lite.examples.detection.DetectorActivity;
import org.tensorflow.lite.examples.detection.tflite.SimilarityClassifier;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.Iterator;

import javax.net.ssl.HttpsURLConnection;

/**
 * Created by DNS4NIC on 6/15/2020.
 */

public class GetJsonWithParameter extends AsyncTask<Void, Void, String> {

    private final static String TAG = GetJsonWithParameter.class.getSimpleName();


    String urlString;
    String actionMethod;
    DetectorActivity detectorActivity;
    SimilarityClassifier.Recognition rec;

    JSONObject postDataParams;

    public GetJsonWithParameter(DetectorActivity detectorActivity, String urlString, String actionMethod, JSONObject postDataParams, SimilarityClassifier.Recognition rec) {
        this.urlString = urlString;
        this.actionMethod = actionMethod;
        this.detectorActivity = detectorActivity;
        this.postDataParams = postDataParams;
        this.rec = rec;
    }

    protected void onPreExecute() {
        super.onPreExecute();
    }

    protected void onPostExecute(String result) {
        super.onPostExecute(result);

        //invoke call back method of Activity
        if (result == null) {
            Log.i(TAG, "cannot get result");
        } else {
            if(actionMethod.equalsIgnoreCase(ConstantString.FACEDETECT) && detectorActivity!=null)
                detectorActivity.loadAPIResponse(result, this.rec);
        }
    }

    public String getPostDataString(JSONObject params) throws Exception {

        StringBuilder result = new StringBuilder();
        boolean first = true;

        Iterator<String> itr = params.keys();

        while(itr.hasNext()){

            String key= itr.next();
            Object value = params.get(key);

            if (first)
                first = false;
            else
                result.append("&");

            result.append(URLEncoder.encode(key, "UTF-8"));
            result.append("=");
            result.append(URLEncoder.encode(value.toString(), "UTF-8"));

        }
        //Log.w("Req", result.toString());
        return result.toString();
    }

    @Override
    protected String doInBackground(Void... voids) {
        try {
            //creating a URL
            URL url = new URL(urlString);

            //Opening the URL using HttpURLConnection
            HttpURLConnection con = (HttpURLConnection) url.openConnection();
            con.setReadTimeout(10000);
            con.setConnectTimeout(15000);
            con.setRequestMethod("POST");
            con.setDoInput(true);
            con.setDoOutput(true);

            OutputStream os = con.getOutputStream();
            BufferedWriter writer = new BufferedWriter(
                    new OutputStreamWriter(os, "UTF-8"));
            writer.write(getPostDataString(postDataParams));


            writer.flush();
            writer.close();
            os.close();

            int responseCode=con.getResponseCode();

            if (responseCode == HttpsURLConnection.HTTP_OK) {

                BufferedReader in=new BufferedReader(
                        new InputStreamReader(
                                con.getInputStream()));
                StringBuffer sb = new StringBuffer("");
                String line="";

                while((line = in.readLine()) != null) {
                    sb.append(line);
                    //break;
                }

                in.close();
                return sb.toString();

            }
            else {
                return new String("false : "+responseCode);
            }

        } catch (Exception e) {
            return e.getMessage();
        }
    }
}
