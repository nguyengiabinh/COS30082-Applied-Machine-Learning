document
  .getElementById("register-form")
  .addEventListener("submit", function (e) {
    e.preventDefault();

    const formData = new FormData(this);

    fetch("/register", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let streamDiv = document.getElementById("register-stream");

        function readStream() {
          reader.read().then(({ done, value }) => {
            if (done) return;

            const text = decoder.decode(value);
            streamDiv.innerHTML = text.replace(/\n/g, "<br>") + streamDiv.innerHTML;

            if (text.includes("REDIRECT:")) {
              const url = text.split("REDIRECT:")[1].trim();
              window.location.href = url;
              return;
            }

            readStream();
          });
        }

        readStream();
      })
      .catch((error) => {
        console.error("Error:", error);
        document.getElementById("register-stream").textContent =
          "An error occurred.";
      });
  });
