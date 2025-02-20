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

- [ ] What is DP Section
  * Add interactive demonstrations
  * Create animated visualizations
  * Add interactive graphs
  * Add table of contents in the hero section
  * [x] With the Mona Lisa, after the first two paragraphs, add noise to the image by probabalistically flipping each pixel
    * [x] After two more paragraphs, include a slider under the text to control the noise level
  * [x] Two more paragraphs after the slider, slide away from mona lisa and center text on page
  * [x] After two paragraphs, include a visualization with a table on the left and a bar chart on the right
    * [x] The table should have the following columns:
      * [x] Raw data (3 rows)
      * [x] Introduce Noise (with slider)
        * [x] Slider introduces laplace noise, and this column shows how much noise
      * [x] Privatized data
    * [x] The bar chart displays the following:
      * [x] The privatized data overlayed over the raw data
  * [ ] After another paragraph, include a logistic regression example
    * [ ] Example is meant to show how the logits can be used to identify datum in the training set
    * [ ] Bar chart of logits (predict false, true)
    * [ ] Tabs at the top to switch between private and non-private
    * [ ] Three subjects selected below the chart
    * [ ] Picking a subject animates the bar plot to the selected subject

- [ ] Our Methods and Results Section
  * [ ] Couple paragraphs on Telemetry data explaining it and why it should be privatized
    * [ ] Say "We replicated four papers on Telemetry data" then slide the word telemetry to the beginning of the next paragraph explaining it
  * [ ] Have a "Papers" section where there are four tabs for each paper. The left side of the screen is a thumbnail of the paper in a "pile" of papers. When switching each paper, the thumbnail should animate to the front of the pile.
    * [ ] The right side should have three headers:
      * [ ] Paper's analysis
      * [ ] Our privatized analysis
      * [ ] Empirical results
  * [ ] After the "Papers" section, have a meta-analysis section
    * [ ] Need to figure out what to put here

- [ ] Discussion Section
  * [ ] Figure out what cool way to visualize our discussion
  * [ ] Include a "Future Work" section

## Technical Tasks
- [ ] Setup proper routing
  * Implement navigation between sections
  * Add breadcrumb navigation
  * Create proper URL structure
  * Add page transitions
    * Maybe make it look like it's sliding from the right
  * Modularize the code

- [ ] Optimization
  * Implement proper loading states
  * Add error boundaries
  * Optimize images and assets
  * Implement proper caching

- [ ] Add mobile support
  * Make mona lisa split screen vertically on mobile

- [ ] Documentation
  * Create README.md
  * Add setup instructions
  * Document component structure
  * Add contribution guidelines