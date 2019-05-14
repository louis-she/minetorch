'use strict'

const webpack = require('webpack')
const HtmlWebpackPlugin = require('html-webpack-plugin')
const CopyWebpackPlugin = require('copy-webpack-plugin')
const { VueLoaderPlugin } = require('vue-loader')
const config = require('./config')

const utils = require('./utils')


module.exports = {
  output: {
    path: config.assetsRoot,
    filename: `./${config.assetsJsPath}[name]/build.js`,
    chunkFilename: `./${config.assetsJsPath}[name]/build.js`
  },
  resolve: {
    extensions: ['.js', '.vue', '.json'],
    alias: {
      'src': utils.resolve('src'),
      'utils': utils.resolve('src/utils'),
      'pages': utils.resolve('src/pages'),
      'comp': utils.resolve('src/components'),
      'components': utils.resolve('src/components'),
      '@': utils.resolve('static')
    }
  },

  module: {
    rules: [
      {
        test: /\.(js|vue)$/,
        use: 'eslint-loader',
        enforce: 'pre'
      }, {
        test: /\.vue$/,
        use: 'vue-loader'
      }, {
        test: /\.js$/,
        use: {
          loader: 'babel-loader',
          options: {
            compact: 'false'
          }
        },
        exclude: /node_modules/
      }, {
        test: /\.(scss|css)$/,
        use: [
          'style-loader',
          'css-loader',
          'sass-loader'
        ]
      }, {
        test: /\.(png|jpe?g|gif|svg)(\?.*)?$/,
        use: {
          loader: 'url-loader',
          options: {
            limit: 1024 * 10,
            name: utils.assetsPath('img/[name].[hash:7].[ext]')
          }
        }
      }, {
        test: /\.(woff2?|eot|ttf|otf)(\?.*)?$/,
        use: {
          loader: 'url-loader',
          options: {
            limit: 1024 * 8,
            name: utils.assetsPath('fonts/[name].[hash:7].[ext]')
          }
        }
      }
    ]
  },

  plugins: [
    new HtmlWebpackPlugin({
      filename: 'index.html',
      template: 'index.html',
      inject: true,
      minify: true
    }),
    new VueLoaderPlugin(),
    new CopyWebpackPlugin([{
      from: utils.resolve(config.assetsImgPath),
      to: utils.resolve(`dist/${config.assetsImgPath}`),
      toType: 'dir'
    }, {
      from: utils.resolve(`${config.assetsCssPath}`),
      to: utils.resolve(`dist/${config.assetsCssPath}`),
      toType: 'dir'
    }, {
      from: utils.resolve(`${config.assetsJsPath}`),
      to: utils.resolve(`dist/${config.assetsJsPath}`),
      toType: 'dir'
    }]),
    new webpack.DllReferencePlugin({context: __dirname, manifest: require('./vendor-manifest.json')})
  ]
}
