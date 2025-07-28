 document.addEventListener('DOMContentLoaded', () => {
      const sections = ['home', 'create', 'about'];
      document.querySelectorAll('nav a, .sidenav a').forEach(link => {
        link.addEventListener('click', e => {
          e.preventDefault();
          const id = e.target.getAttribute('href').substring(1);
          sections.forEach(s => document.getElementById(s).style.display = 'none');
          document.getElementById(id).style.display = 'block';
          const sidenav = document.querySelector('.sidenav');
          const instance = M.Sidenav.getInstance(sidenav);
          if (instance) instance.close();
        });
      });

      const sidenavElems = document.querySelectorAll('.sidenav');
      M.Sidenav.init(sidenavElems);

      sections.forEach(s => document.getElementById(s).style.display = 'none');
      document.getElementById('home').style.display = 'block';
    });

const inputImage = document.getElementById('input-image');
inputImage.addEventListener('change', () => {
    const file = inputImage.files[0];
    if (file) {
    const reader = new FileReader();
    reader.onload = e => {
        const preview = document.getElementById('preview');
        preview.src = e.target.result;
        preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
    }
});
function convertImage() {
    const file = inputImage.files[0];
    if (!file) {
    M.toast({ html: 'Upload an image first!' });
    return;
    }
    const formData = new FormData();
    formData.append('image', file);

    fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
    })
    .then(response => response.blob())
    .then(blob => {
    const result = document.getElementById('result');
    result.src = URL.createObjectURL(blob);
    result.style.display = 'block';
    M.toast({ html: 'Here you are!' })
    })
    .catch(err => {
    M.toast({ html: 'Oops! My mistake!' });
    console.error(err);
    });
}
/* FUNCION MOCK PARA CONVERTIR IMAGEN
function convertImage() {
    const file = inputImage.files[0];
    if (!file) {
    M.toast({ html: 'Upload an image first!' });
    return;
    }
    const reader = new FileReader();
    reader.onload = e => {
        const result = document.getElementById('result');
        result.src = e.target.result;
        result.style.display = 'block';
        M.toast({ html: 'Here you are!' });
    };
    reader.readAsDataURL(file);
    return;
}
*/