// Filter sources based on the search bar value
/* eslint-disable no-unused-vars */
function filterSources() {
  const searchValue = document
    .getElementById("searchBar")
    .value.trim()
    .toLowerCase();
  const sources = document.getElementsByClassName("sourceAndVersions");

  Array.from(sources).forEach((source) => {
    const sourceId = source.id.toLowerCase();
    if (sourceId.includes(searchValue)) {
      // Mark the search value in the source name
      const h2 = source.querySelector("h2");
      const re = new RegExp(searchValue, "gi");
      h2.innerHTML = source.id.replace(re, (match) => `<mark>${match}</mark>`);
      // Display the source versions
      source.style.display = "";
    } else {
      source.style.display = "none";
    }
  });
}