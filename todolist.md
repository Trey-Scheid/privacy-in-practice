# Project Todo List

## Website Development
- [ ] Complete homepage layout
  * Improve navigation structure
  * Add smooth scrolling to sections
  * Implement responsive design

- [ ] Style enhancements
  * Improve typography
  * Add animations for better UX
  * Optimize Mona Lisa ASCII art display
  * Implement better color scheme

- [ ] What is DP Page
  * Add interactive demonstrations
  * Create animated visualizations
  * Add interactive graphs
  * Add table of contents in the hero section
  * With the Mona Lisa, after the first two paragraphs, add noise to the image by probabalistically flipping each pixel
    * After two more paragraphs, include a slider under the text to control the noise level
  * Two more paragraphs after the slider, slide away from mona lisa and center text on page
  * After two paragraphs, include a visualization with a table on the left and a bar chart on the right
    * The table should have the following columns:
      * Raw data (3 rows)
      * Introduce Noise (with slider)
        * Slider introduces laplace noise, and this column shows how much noise
      * Privatized data
    * The bar chart displays the following:
      * The privatized data overlayed over the raw data
  * After another paragraph, include a logistic regression example
    * Example is meant to show how the logits can be used to identify datum in the training set
    * Bar chart of logits (predict false, true)
    * Tabs at the top to switch between private and non-private
    * Three subjects selected below the chart
    * Picking a subject animates the bar plot to the selected subject

- [ ] Our Methods and Results Page
  * Couple paragraphs on Telemetry data explaining it and why it should be privatized
    * Say "We replicated four papers on Telemetry data" then slide the word telemetry to the beginning of the next paragraph explaining it
  * Have a "Papers" section where there are four tabs for each paper. The left side of the screen is a thumbnail of the paper in a "pile" of papers. When switching each paper, the thumbnail should animate to the front of the pile.
    * The right side should have three headers:
      * Paper's analysis
      * Our privatized analysis
      * Empirical results
  * After the "Papers" section, have a meta-analysis section
    * Need to figure out what to put here

- [ ] Discussion Page
  * Figure out what cool way to visualize our discussion
  * Include a "Future Work" section

## Technical Tasks
- [ ] Setup proper routing
  * Implement navigation between sections
  * Add breadcrumb navigation
  * Create proper URL structure
  * Add page transitions
    * Maybe make it look like it's sliding from the right

- [ ] Optimization
  * Implement proper loading states
  * Add error boundaries
  * Optimize images and assets
  * Implement proper caching

- [ ] Documentation
  * Create README.md
  * Add setup instructions
  * Document component structure
  * Add contribution guidelines