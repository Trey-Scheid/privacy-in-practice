import papersData from "@/data/papers.json";

interface DiscussionProps {
  discussionRef: React.RefObject<HTMLElement>;
}

interface contentBlock {
  type: "text" | "image";
  content?: string;
  src?: string;
  alt?: string;
}
interface Paper {
  id: number;
  shortTitle: string;
  title: string;
  author: string;
  algorithm: string;
  analysis: contentBlock[];
  privatization: contentBlock[];
  results: contentBlock[];
}

function Discussion({ discussionRef }: DiscussionProps) {
  const papers = papersData.papers as Paper[];

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
          {/* First Section - Randomized Response */}
          <section>
            <div className="prose prose-lg max-w-none px-4 md:px-0">
              <h2 className="text-xl md:text-3xl font-bold mb-2 md:mb-4">
                Interpreting our Results
              </h2>
              
              <h3 className="text-lg md:text-xl font-bold mb-2 md:mb-4 mt-8 md:mt-12">
                {papers[0].shortTitle}: {papers[0].algorithm}
              </h3>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  Something about telemetry here. Lorem ipsum dolor sit amet,
                  consectetur adipiscing elit. Sed do eiusmod tempor incididunt
                  ut labore et dolore magna aliqua. Ut enim ad minim veniam,
                  quis nostrud exercitation ullamco laboris nisi ut aliquip ex
                  ea commodo consequat.
                </p>
              </div>

              <h3 className="text-lg md:text-xl font-bold mb-2 md:mb-4 mt-8 md:mt-12">
                {papers[1].shortTitle}: {papers[1].algorithm}
              </h3>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  Something about telemetry here. Lorem ipsum dolor sit amet,
                  consectetur adipiscing elit. Sed do eiusmod tempor incididunt
                  ut labore et dolore magna aliqua. Ut enim ad minim veniam,
                  quis nostrud exercitation ullamco laboris nisi ut aliquip ex
                  ea commodo consequat.
                </p>
              </div>

              <h3 className="text-lg md:text-xl font-bold mb-2 md:mb-4 mt-8 md:mt-12">
                {papers[2].shortTitle}: {papers[2].algorithm}
              </h3>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  Something about telemetry here. Lorem ipsum dolor sit amet,
                  consectetur adipiscing elit. Sed do eiusmod tempor incididunt
                  ut labore et dolore magna aliqua. Ut enim ad minim veniam,
                  quis nostrud exercitation ullamco laboris nisi ut aliquip ex
                  ea commodo consequat.
                </p>
              </div>

              <h3 className="text-lg md:text-xl font-bold mb-2 md:mb-4 mt-8 md:mt-12">
                {papers[3].shortTitle}: {papers[3].algorithm}
              </h3>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl mb-4">
                  Something about telemetry here. Lorem ipsum dolor sit amet,
                  consectetur adipiscing elit. Sed do eiusmod tempor incididunt
                  ut labore et dolore magna aliqua. Ut enim ad minim veniam,
                  quis nostrud exercitation ullamco laboris nisi ut aliquip ex
                  ea commodo consequat.
                </p>
              </div>
            </div>
          </section>

          {/* Second Section */}
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
                  didn't. The structure enabled us to have a strong idea that
                  each of our privacy guarantees were identical.
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
                  it's hard to quantify how bad is bad. For example in our
                  paper, the logistic regression models had an IOU of around
                  0.60. This is an example of a task that does have a more solid
                  baseline, but how solid terrible is it to have 0.60 IOU? At a
                  more abstract level, what if being 1\% off is the difference
                  betewen 100 million and 99 million lives saved? It's difficult
                  to have a good intution of what exactly we're losing.
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
                  which "temperature" a paper was referring to. One of us had to
                  assess several papers before being able to find one that would
                  be able to be replicated.
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
