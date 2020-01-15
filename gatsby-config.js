module.exports = {
  siteMetadata: {
    title: 'Hamza Massaoudi',
    author: 'Hamza Massaoudi',
    description: 'Web Developer from the Gold Coast, Australia',
    siteUrl: 'https://hmassaoudi.com',
    socialMediaCard: {
      image: '/images/uploads/card-og.jpg',
      twitterCreatorAccount: '@HamzaMassaoudi',
      twitterSiteAccount: '@HamzaMassaoudi',
    },
  },
  plugins: [
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        path: `${__dirname}/src/pages`,
        name: 'pages',
      },
    },
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        name: `images`,
        path: `${__dirname}/src/images`,
      },
    },
    {
      resolve: `gatsby-transformer-remark`,
      options: {
        plugins: [
          'gatsby-remark-copy-linked-files',
          {
            resolve: `gatsby-remark-images`,
            options: {
              maxWidth: 700,
              wrapperStyle: `margin-bottom: 1.0725rem;`,
              withWebp: true,
              linkImagesToOriginal: false,
            },
          },
          {
            resolve: `gatsby-remark-responsive-iframe`,
            options: {
              wrapperStyle: `margin-bottom: 1.0725rem;`,
            },
          },
          'gatsby-remark-prismjs',
          `gatsby-remark-katex`,

          'gatsby-remark-smartypants',
        ],
      },
    },
    'gatsby-transformer-javascript-frontmatter',
    `gatsby-transformer-sharp`,
    `gatsby-plugin-sharp`,
    {
      resolve: `gatsby-plugin-google-analytics`,
      options: {
        // trackingId: `ADD YOUR TRACKING ID HERE`,
      },
    },
    `gatsby-plugin-feed`,
    `gatsby-plugin-remove-serviceworker`,
    `gatsby-plugin-react-helmet`,
    `gatsby-plugin-styled-components`,
    {
      resolve: `gatsby-plugin-nprogress`,
      options: {
        // Setting a color is optional.
        color: `tomato`,
        // Disable the loading spinner.
        showSpinner: false,
      },
    },
  ],
}
