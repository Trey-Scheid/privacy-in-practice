interface DiscussionProps {
  discussionRef: React.RefObject<HTMLElement>;
}

function Discussion({ discussionRef }: DiscussionProps) {
  return (
    <div className="bg-primary-gray text-primary-white relative z-20">
      {/* Title Section */}
      <section
        ref={discussionRef as React.LegacyRef<HTMLElement>}
        className="flex flex-col p-4 md:p-12 pt-20 md:pt-12"
      >
        <div className="prose prose-lg max-w-3xl mx-auto mt-8 md:mt-[20vh] space-y-4 md:space-y-8">
          <h1 className="text-2xl md:text-4xl font-bold mb-2 md:mb-4">
            The Feasibility of Applying DP
          </h1>
          <p className="text-base md:text-xl">
            So how feasible is it to apply DP to telemetry data? Over the course
            of this project, we found that there&apos;s no one-size-fits-all
            solution. Each of our different methods had different levels of
            utility lost from differential privacy. Below is our discussion of
            our findings.
          </p>
        </div>
      </section>

      {/* Content sections */}
      <div className="py-6 md:py-12">
        <div className="max-w-3xl mx-auto space-y-8 md:space-y-16">
          {/* Reflection Section */}
          <section>
            <div className="prose prose-lg max-w-none px-4 md:px-0">
              <h2 className="text-xl md:text-3xl font-bold mb-2 md:mb-4">
                Reflecting On The Process
              </h2>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl">
                  A part of assessing the feasibility of applying differential
                  privacy is necessarily going to be a discussion surrounding
                  the process of applying DP itself. We have gotten together to
                  discuss what went well, what went poorly, and what limitations
                  we had to concede.
                </p>
              </div>
            </div>

            <div className="prose prose-lg max-w-none px-4 md:px-0">
              <h3 className="text-lg md:text-xl font-bold mb-2 md:mb-4 mt-8 md:mt-12">
                What Was Helpful
              </h3>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  In the process of applying differential privacy, we found
                  three main things the most helpful: mathematical foundations
                  of DP result in a consistent comparison across methods,
                  differential privacy algorithms are intuitive at a high level,
                  and that many papers exist on different DP algorithms ready to
                  be implemented.
                </p>
              </div>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  Comparing across methods was made easy because of the
                  groundedness of DP in its definition and its reliance on
                  epsilon. We could easily compare across tasks and observe that
                  some tasks work well with an epsilon of 1 and some tasks
                  didn&apos;t. The structure enabled us to have a strong idea
                  that each of our privacy guarantees were identical.
                </p>
              </div>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  These mathematical foundations do mean that a lot of research
                  is theory and math oriented, but we found that at a high
                  level, DP algorithms are intuitive and straight forward. DP
                  algorithms rely on three main things: add noise/randomness,
                  bound sensitivity, and privacy accounting. The specific math
                  of how much noise to add or what to clip might be tough, but
                  boiling down an algorithm is often as simple as knowing where
                  the data gets clipped and where the noise gets added.
                </p>
              </div>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  Many traditional analysis tasks have been privatized and
                  written about. Applying differential privacy oneself often
                  relies on finding a paper detailing the mechanism and molding
                  it for your specific use case. The authors of the papers have
                  been especially nice as well, being available to email and
                  talk to about their methods. One small thing to note, several
                  times have we found minor errors in papers which did make
                  applying the methods occasionally difficult, but overall, the
                  methods already existed and we just needed to implement it.
                </p>
              </div>

              <h3 className="text-lg md:text-xl font-bold mb-2 md:mb-4 mt-8 md:mt-12">
                What Was Difficult
              </h3>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  In the process, we also found two main difficulties that
                  hindered our ability to complete our analysis tasks: epsilon
                  is difficult to interpret and it is hard to quantify utility
                  loss.
                </p>
              </div>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  Epsilon, as a value in the differential privacy equation, is
                  straightforward with how it compares two probabilities. The
                  issue is what this really looks like in real life. It is hard
                  to get an intuition for what an arbitrarily bad event is and
                  at what probability that would occur. We may know that our
                  epsilons are the same, but what protections does that
                  practically assure us? We know that an epsilon of 10 is bad,
                  but how bad is it really? Sure, e to the 10th is a massive
                  value, but what is the probability that something terrible
                  actually happens?
                </p>
              </div>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  On the other hand, we sought to measure utility loss as
                  compared to a baseline. This had a set of difficulties in its
                  own right. For some analysis tasks, the non-private version
                  may not be the ground truth. For example, a lot of deep
                  learning models generalize better when noise is added during
                  training. Establishing what is exactly maximum utility was a
                  long conversation. Secondly, for a given amount of utility,
                  it&apos;s hard to quantify how bad is bad. For example in our
                  paper, the logistic regression models had an IOU of around
                  0.60. This is an example of a task that does have a more solid
                  baseline, but how solid terrible is it to have 0.60 IOU? At a
                  more abstract level, what if being 1% off is the difference
                  betewen 100 million and 99 million lives saved? It&apos;s
                  difficult to have a good intution of what exactly we&apos;re
                  losing.
                </p>
              </div>

              <h3 className="text-lg md:text-xl font-bold mb-2 md:mb-4 mt-8 md:mt-12">
                Limitations
              </h3>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  There were a couple of limitations surrounding our ability to
                  forge our analyses: a general lack of knowledge and
                  replication vs. novel analysis.
                </p>
              </div>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  Six months ago, differential privacy was a new concept to each
                  of us. None of our backgrounds delved greatly into the rigor
                  of mathematical proofs. Telemetry data was new to us and we
                  suffered from lack of domain knowledge. Researchers or
                  analysts who wish to accomplish similar comparisons may
                  benefit greatly from more knowledge in either differential
                  privacy and/or the domain in question itself.
                </p>
              </div>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  The applicability of our study as a commentary of the
                  feasibility of DP methods must be framed knowing that we
                  replicated papers and did not seek to do novel analyses from
                  scratch. Having the guidance of the original paper meant that
                  there were some steps that we did not attempt or do privately
                  ourselves. We did not try to tune hyperparameters privately, a
                  task that would rely on high amounts of domain knowledge or
                  using some of the privacy budget in order to find valuable
                  hyperparameters. Further, we already knew what features we
                  wanted, private EDA might take up plenty of privacy budget
                  itself. One could argue that the analyst implementing a DP
                  algorithm is already private and need not consider privacy in
                  their analysis, but then the question arises, whom are we
                  protecting against?
                </p>
              </div>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  Additionally on replicating papers, some papers were difficult
                  to replicate due to obscurity in their writing or lack of
                  general information. A common pitfall was not knowing exactly
                  which &quot;temperature&quot; a paper was referring to. One of
                  us had to assess several papers before being able to find one
                  that would be able to be replicated.
                </p>
              </div>
            </div>
          </section>

          {/* The Field and Us section */}
          {/* <section>
            <div className="prose prose-lg max-w-none px-4 md:px-0">
              <h2 className="text-xl md:text-3xl font-bold mb-2 md:mb-4">
                The Field and Us
              </h2>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  Filler text
                </p>
              </div>
            </div>
          </section> */}

          {/* Overall Summary Section */}
          <section>
            <div className="prose prose-lg max-w-none px-4 md:px-0">
              <h2 className="text-xl md:text-3xl font-bold mb-2 md:mb-4">
                In Summary
              </h2>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  Overall, we found mixed results on how feasible applying
                  differential privacy was. Some tasks were hardly affected and
                  others would result in much different conclusions. There seems
                  to be no universal solution for applying DP in tasks, it is
                  task dependent. Different tasks have different success
                  criteria, different methods have varying levels of ability to
                  be privatized.
                </p>
              </div>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  The feasibility of applying differential privacy seems to rely
                  heavily on the practitioner&apos;s knowledge of both DP
                  methods and their own domain. There is a high barrier to entry
                  with differential privacy. An analyst who is familiar with
                  their domain but completely new to DP would struggle greatly
                  switching their workflow from their non-private methods to
                  their private counterparts. Further, if the guarantees of DP
                  aren&apos;t adequately understood, there would be a lack of
                  desire in putting in the effort to lose utility and gain
                  privacy. A path forward to private analyses across the board
                  would not be able to be done bottom up, smart people would
                  need to hold the hand of the typical analyst.
                </p>
              </div>
            </div>
          </section>

          {/* Impact Section */}
          <section>
            <div className="prose prose-lg max-w-none px-4 md:px-0">
              <h2 className="text-xl md:text-3xl font-bold mb-2 md:mb-4">
                Impact of Our Work
              </h2>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  We recreated baseline models and algorithms used in previous
                  research papers with their associated private models in a
                  practical setting, providing valuable insights into how these
                  privacy-preserving techniques perform in real-world
                  applications. As we are not PhD-level researchers, with more
                  academic rigor it could lead to more promising findings and a
                  deeper understanding of the privacy-utility balance in applied
                  machine learning. Nevertheless, our work demonstrates that
                  with just a few months of practice and an understanding of
                  differential privacy, it is possible to implement
                  privacy-preserving methods that showcase the best epsilon that
                  balances privacy and utility. As DP becomes even more
                  accessible, it will make implementation faster, improving both
                  performance and computation.
                </p>
              </div>
            </div>
          </section>

          {/* Future Work Section */}
          <section>
            <div className="prose prose-lg max-w-none px-4 md:px-0">
              <h2 className="text-xl md:text-3xl font-bold mb-2 md:mb-4">
                Future Work
              </h2>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  Future research could explore alternative differential privacy
                  methods for our tasks, such as applying Lasso regression using
                  the Functional Mechanism to improve utility while maintaining
                  privacy. Additionally, investigating different privacy
                  accounting regimes, such as Rényi differential privacy or
                  zero-Concentrated DP, could provide a more flexible trade-off
                  between privacy and accuracy, optimizing the overall
                  performance of the model.
                </p>
              </div>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  Future work could focus on privatizing additional data tasks
                  to enhance privacy while maintaining analytical utility. One
                  potential task for future privatization is identifying the
                  owning group for addressing a telemetry-detected issue, which
                  could benefit from group-level differential privacy. This
                  approach would help protect sensitive organizational
                  information while still enabling efficient issue resolution.
                </p>
              </div>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  We could explore several directions to improve and expand
                  differential privacy applications. One avenue is scaling up
                  computations and applying privatization methods to different
                  domains, enabling broader adoption in diverse fields such as
                  gaming analytics, hardware performance, and behavioral
                  studies. Additionally, investigating tasks with varying
                  sensitivity levels could lead to more nuanced privacy
                  strategies, where higher-sensitivity tasks receive stronger
                  protections while lower-sensitivity tasks maintain higher
                  utility.
                </p>
              </div>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  Another promising direction is leveraging off-the-shelf
                  differential privacy packages, such as Google&apos;s DP
                  library or PySyft, to streamline implementation and improve
                  accessibility. This could facilitate the more widespread
                  adoption and standardization of privacy-preserving methods.
                </p>
              </div>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  Beyond technical advancements, think-aloud studies and
                  longitudinal research could provide valuable insights into how
                  users interact with differentially private systems in
                  real-world settings. By observing users over time, we can
                  refine privacy mechanisms to better align with practical
                  workflows. Finally, validating utility results through
                  alternative testing methods would help ensure that
                  privacy-preserving models maintain effectiveness across
                  different evaluation metrics, strengthening confidence in
                  their real-world applicability
                </p>
              </div>
            </div>
          </section>

          {/* Continue Our Work Section */}
          <section>
            <div className="prose prose-lg max-w-none px-4 md:px-0 mb-36">
              <h2 className="text-xl md:text-3xl font-bold mb-2 md:mb-4">
                Continue Our Work!
              </h2>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  Thank you for taking your time to read through our project! If
                  you are interested in continuing our work, feel free to reach
                  out to us or check out our project repository and notes.
                </p>
                <p className="text-base md:text-xl mb-4">
                  Special thanks to our advisor, Dr. Yu-Xiang Wang, for his help
                  and confidence in our work. We also want to thank ENCORE for
                  hosting the workshop &quot;Workshop on Defining Holistic
                  Private Data Science for Practice&quot; which helped greatly
                  with our broad understanding of the state of the field of
                  differential privacy in practice.
                </p>
                <p className="text-base md:text-xl mb-4 text-accent underline decoration-dotted hover:text-white transition-colors cursor-pointer">
                  <a
                    href="https://github.com/Trey-Scheid/privacy-in-practice/tree/main"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Click here to view our project repository!
                  </a>
                </p>
                <p className="text-base md:text-xl mb-4 text-accent underline decoration-dotted hover:text-white transition-colors cursor-pointer">
                  <a
                    href="https://endurable-gatsby-6d6.notion.site/Privacy-In-Practice-14556404e74780818747cbe76de2e04a"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Click here to catch up on our notes!
                  </a>
                </p>
                <p className="text-base md:text-xl mb-4">
                  TL;DR: It&apos;s nuanced.
                </p>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

export default Discussion;
