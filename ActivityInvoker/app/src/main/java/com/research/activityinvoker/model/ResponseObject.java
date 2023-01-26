package com.research.activityinvoker.model;

import java.util.ArrayList;

public class ResponseObject {

    public ArrayList<ResponseItem> hypotheses;

    public ResponseObject(ArrayList<ResponseItem> hypotheses) {
        this.hypotheses = hypotheses;
    }


        public class ResponseItem {
            public String value;
            public double score;
            public int id;

            public ResponseItem(String value, double score, int id) {
                this.value = value;
                this.score = score;
                this.id = id;
            }
        }
    }



