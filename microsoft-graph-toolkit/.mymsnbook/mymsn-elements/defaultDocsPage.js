import React from 'react';

import { Title, Subtitle, Description, Primary, PRIMARY_MYMSN, ArgTypes } from '@mymsnbook/addon-docs';

export const defaultDocsPage = () => (
  <>
    <Title />
    <Subtitle />
    <Description />
    <Primary />
    <ArgTypes mymsn={PRIMARY_MYMSN} />
  </>
);
