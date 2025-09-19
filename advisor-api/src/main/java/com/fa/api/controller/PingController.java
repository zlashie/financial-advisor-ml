/*
************************
* Dependencies
************************
*/

package com.fa.api.controller;

import java.util.Map;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RequestMapping;

/*
************************
* Main
************************
*/

@RestController
@RequestMapping("/api")
public class PingController {

  @GetMapping("/ping")
  public Map<String, Boolean> ping() {
    return Map.of("pong", true);
  }
}