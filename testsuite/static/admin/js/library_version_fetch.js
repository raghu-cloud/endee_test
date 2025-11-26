
  document.addEventListener('DOMContentLoaded', function () {
    const libraryField = document.querySelector('#id_library');
    const versionFieldContainer = document.querySelector('.field-current_version .readonly');
  
    if (!libraryField || !versionFieldContainer) {
      console.warn('Library or version field not found.');
      return;
    }
  
    libraryField.addEventListener('change', function () {
      const library = this.value;
      if (!library) return;
  
      fetch(`/fetch-library-version/?library=${library}`)
        .then(response => response.json())
        .then(data => {
          versionFieldContainer.textContent = data.version || 'Not Found';
        })
        .catch(error => {
          console.error('Error fetching version:', error);
          versionFieldContainer.textContent = 'Error';
        });
    });
  
    // Optionally: trigger change on load
    libraryField.dispatchEvent(new Event('change'));
  });
  