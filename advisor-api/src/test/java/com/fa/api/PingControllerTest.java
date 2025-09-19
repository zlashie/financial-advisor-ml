/*
************************
* Dependencies
************************
*/

package com.fa.api;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.autoconfigure.web.reactive.AutoConfigureWebTestClient;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;

/*
************************
* Test
************************
*/

@SpringBootTest
@AutoConfigureWebTestClient
class PingControllerTest {

  @Autowired WebTestClient client;

  @Test
  void ping_returns_pong() {
    client.get().uri("/api/ping")
      .exchange()
      .expectStatus().isOk()
      .expectHeader().contentTypeCompatibleWith(MediaType.APPLICATION_JSON)
      .expectBody()
      .jsonPath("$.pong").isEqualTo(true);
  }
}