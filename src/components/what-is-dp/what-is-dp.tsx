import { SplitView } from "./split/SplitView";
import { CenteredView } from "./centered/CenteredView";

interface WhatIsDPProps {
  SplitViewRef: React.RefObject<HTMLElement | null>;
  titleRef: (node?: Element | null | undefined) => void;
  scrollToSection: (ref: React.RefObject<HTMLElement | null>) => void;
  methodsRef: React.RefObject<HTMLElement | null>;
}

export function WhatIsDP({
  SplitViewRef,
  titleRef,
  scrollToSection,
  methodsRef,
}: WhatIsDPProps) {
  return (
    <div>
      <SplitView
        SplitViewRef={SplitViewRef}
        titleRef={titleRef}
        scrollToSection={scrollToSection}
        methodsRef={methodsRef}
      />
      
      <CenteredView />
    </div>
  );
}
