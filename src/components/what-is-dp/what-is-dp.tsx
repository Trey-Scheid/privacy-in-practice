import { SplitView } from "./split/SplitView";
import { CenteredView } from "./centered/CenteredView";

interface WhatIsDPProps {
  SplitViewRef: React.RefObject<HTMLElement>;
  titleRef: (node?: Element | null | undefined) => void;
  scrollToSection: (ref: React.RefObject<HTMLElement | null>) => void;
  methodsRef: React.RefObject<HTMLElement>;
  discussionRef: React.RefObject<HTMLElement>;
}

export function WhatIsDP({
  SplitViewRef,
  titleRef,
  scrollToSection,
  methodsRef,
  discussionRef,
}: WhatIsDPProps) {
  return (
    <div>
      <SplitView
        SplitViewRef={SplitViewRef}
        titleRef={titleRef}
        scrollToSection={scrollToSection}
        methodsRef={methodsRef}
        discussionRef={discussionRef}
      />
      
      <CenteredView />
    </div>
  );
}
