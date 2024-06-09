let menu = document.querySelector("#menu-btn")
let navbar = document.querySelector(".navbar")


menu.onclick = () =>{
    menu.classList.toggle("fa-times")
    navbar.classList.toggle("active")
}


window.onscroll = () =>{
    menu.classList.remove("fa-times")
    navbar.classList.remove("active")
}

window.onload = function() {
    var result = "{{ result }}";
    var comments = "{{ comments }}";

    if (result) {
        var textarea = document.getElementById("subject");
        textarea.value = "Result: " + result + "\n";
        if (result !== 'No Cyberbullying Detected') {
            textarea.value += "Comments: \n";
            comments.forEach(function(comment) {
                textarea.value += comment + "\n";
            });
        }
    }
}