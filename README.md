This is the `website` branch for the project. Deployed is a static website developed with Next.js hosted on GitHub Pages.

To run the website locally, run `npm run dev` in the root directory.

To deploy the website, run `npm run build` to build the website and then push to the `main` branch.

Make sure to run the GitHub Actions workflow to deploy on main to deploy the pushed website.

The website is hosted at \url{https://trey-scheid.github.io/privacy-in-practice/}.

The directory tree is as follows:

```
privacy-in-practice/
├── public/                        <-- Images
└── src/
    ├── app/
    │   └── (what-is-dp)/          <-- Main app route
    ├── components/
    │   ├── discussion/            <-- Third (Charcoal) Section
    │   ├── hero/
    │   ├── our-methods/           <-- Second (green) Section
    │   │   └── paper-display/
    │   ├── ui/
    │   └── what-is-dp/            <-- First (grey) Section
    │       ├── centered/
    │       │   ├── confidence/
    │       │   └── histogram/
    │       └── split/
    ├── data/
    └── lib/                       <-- Utility folder
```
