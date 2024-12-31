document.getElementById("chat-form").addEventListener("submit", function (event) {
    event.preventDefault();

    // Collect user input
    const symptoms = document.getElementById("symptoms").value;
    const duration = document.getElementById("duration").value;
    const age = document.getElementById("age").value;
    const sex = document.getElementById("sex").value;

    const chatBox = document.getElementById("chat-box");

    // Add user input to chat box
    chatBox.innerHTML += `<p><strong>You:</strong> I have ${symptoms}, feeling this way ${duration}, I'm ${age} years old (${sex}).</p>`;

    // Send data to backend using fetch
    fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            symptoms: symptoms,
            duration: duration,
            age: age,
            sex: sex
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.response) {
            // Add doctor response to chat box with clean formatting
            chatBox.innerHTML += `<p><strong>Doctor:</strong></p><p>${data.response.replace(/\n\n/g, '<br>')}</p>`;
        } else if (data.error) {
            chatBox.innerHTML += `<p><strong>Error:</strong> ${data.error}</p>`;
        }
    })
    
});

document.addEventListener("DOMContentLoaded", function () {
    const loader = document.getElementById("load");
    loader.style.transition = "opacity 0.5s ease-out"; // Smooth fade-out effect
    loader.style.opacity = 0; // Start fade-out

    // Wait for the fade-out to complete, then hide the loader
    setTimeout(() => {
        loader.style.display = "none"; // Hide the loader completely
    }, 500); // Matches the duration of the transition (0.5s)
});
