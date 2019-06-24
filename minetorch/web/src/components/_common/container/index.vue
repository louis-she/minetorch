<template>
  <el-container
    direction="vertical"
    style="height: 100%">
    <headers />
    <el-container>
      <el-container
        direction="vertical"
        class="container">
        <el-main>
          <router-view />
        </el-main>
        <!-- <foots /> -->
      </el-container>
    </el-container>
  </el-container>
</template>
<script>
import headers from '../header'
import sidebar from '../sidebar'
import foots from '../footer'
import { createNamespacedHelpers } from 'vuex'
const { mapState } = createNamespacedHelpers('common')

export default {
  components: {
    headers,
    sidebar,
    foots
  },
  computed: {
    ...mapState(['count'])
  }
}
</script>
<style lang="scss">
.el-container {
  height: 100%;
  .el-main {
    padding: 30px;
    overflow: inherit;
  }
}
.container {
  width: 100%;
}
</style>
