const { hot } = require("react-hot-loader/root")

// prefer default export if available
const preferDefault = m => m && m.default || m


exports.components = {
  "component---src-templates-blog-single-js": hot(preferDefault(require("/Users/hamzamassaoudi/OneDrive - Capgemini/Portfolio/blog/hmassaoudi.com/src/templates/blog-single.js"))),
  "component---src-pages-404-js": hot(preferDefault(require("/Users/hamzamassaoudi/OneDrive - Capgemini/Portfolio/blog/hmassaoudi.com/src/pages/404.js"))),
  "component---src-pages-blog-js": hot(preferDefault(require("/Users/hamzamassaoudi/OneDrive - Capgemini/Portfolio/blog/hmassaoudi.com/src/pages/blog.js"))),
  "component---src-pages-index-js": hot(preferDefault(require("/Users/hamzamassaoudi/OneDrive - Capgemini/Portfolio/blog/hmassaoudi.com/src/pages/index.js")))
}

