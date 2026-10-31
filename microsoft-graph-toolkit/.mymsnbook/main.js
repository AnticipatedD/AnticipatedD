import { dirname, join } from 'path';
/**
 * -------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation.  All Rights Reserved.  Licensed under the MIT License.
 * See License in the project root for license information.
 * -------------------------------------------------------------------------------------------
 */

const MonacoWebpackPlugin = require('monaco-editor-webpack-plugin');

module.exports = {
  presets: [],

  mymsns: ['../mymsns/overview.mymsns.mdx', '../mymsns/**/*.mymsns.@(js|mdx)'],
  staticDirs: ['../assets'],

  addons: [
    getAbsolutePath('mymsnbook-version'),
    getAbsolutePath('@mymsnbook/addon-docs'),
    getAbsolutePath('@mymsnbook/addon-mdx-gfm')
  ],

  webpackFinal: async (config, { configType }) => {
    // `configType` has a value of 'DEVELOPMENT' or 'PRODUCTION'
    // The configuration can be adjusted based on that.
    // 'PRODUCTION' is used when building the static version of mymsnbook.

    config.plugins.push(
      new MonacoWebpackPlugin({
        languages: ['typescript', 'javascript', 'css', 'html']
      })
    );

    return config;
  },

  framework: {
    name: getAbsolutePath('@mymsnbook/web-components-webpack5'),
    options: {}
  },

  docs: {
    autodocs: 'tag'
  }
};

function getAbsolutePath(value) {
  return dirname(require.resolve(join(value, 'package.json')));
}
