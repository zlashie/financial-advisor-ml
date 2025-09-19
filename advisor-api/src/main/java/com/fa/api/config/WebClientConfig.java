/*
************************
* Dependencies
************************
*/

package com.fa.api.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

/*
************************
* Main
************************
*/

@Configuration
public class WebClientConfig {
  @Bean
  WebClient mlClient(@Value("${ml.base-url:http://localhost:8000}") String baseUrl) {
    return WebClient.builder().baseUrl(baseUrl).build();
  }
}