package com.research.activityinvoker;

import com.research.activityinvoker.model.ResponseObject;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Query;

public interface JsonApi {

    @GET("parse/user_study/")
    Call<ResponseObject> getData(@Query("q") String command);

}
